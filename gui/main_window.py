from PySide6 import QtCore, QtWidgets

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from dlgo import GameState, Player, compute_game_result
from dlgo.goboard import Move
from dlgo.scoring import evaluate_territory

from .board_widget import GoBoardWidget


class AIMoveWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(str, object)

    def __init__(self, agent, game_state, state_key):
        super().__init__()
        self._agent = agent
        self._game_state = game_state
        self._state_key = state_key

    @QtCore.Slot()
    def run(self):
        try:
            move = self._agent.select_move(self._game_state)
            self.finished.emit(move, self._state_key)
        except Exception as exc:
            self.failed.emit(str(exc), self._state_key)


class GoMainWindow(QtWidgets.QMainWindow):
    def __init__(self, komi_override=None):
        super().__init__()
        self.setWindowTitle("Go AI Arena")
        self.resize(1220, 820)

        self.game_state = None
        self.history = []
        self.player_kinds = {}
        self.player_agents = {}
        self._busy = False
        self._ai_thread = None
        self._ai_worker = None
        self._pending_state_key = None

        self._forced_game_over = False
        self._forced_winner = None
        self._forced_reason = ""
        self._forced_result_text = ""
        self._move_limit = 50
        self._form_rows = {}
        self._komi_override = komi_override

        self._build_ui()
        self._install_style()
        self._sync_mode_visibility()
        self._sync_agent_options()
        self.start_new_game()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        self.board_widget = GoBoardWidget()
        self.board_widget.point_clicked.connect(self.on_human_click)
        root.addWidget(self.board_widget, stretch=3)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(340)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        root.addWidget(side, stretch=2)

        side_layout.addWidget(self._build_setup_group())
        side_layout.addWidget(self._build_action_group())
        side_layout.addWidget(self._build_status_group())
        side_layout.addWidget(self._build_log_group(), stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_setup_group(self):
        group = QtWidgets.QGroupBox("Game Setup")
        self.setup_layout = QtWidgets.QFormLayout(group)
        self.setup_layout.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.size_combo = QtWidgets.QComboBox()
        self.size_combo.addItems(["5", "7", "9"])
        self.size_combo.setCurrentText("5")
        self._add_form_row("size", "Board Size", self.size_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Human vs AI", "human_ai")
        self.mode_combo.addItem("AI vs AI", "ai_ai")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_visibility)
        self.mode_combo.currentIndexChanged.connect(self._sync_agent_options)
        self._add_form_row("mode", "Mode", self.mode_combo)

        self.move_limit_guard_checkbox = QtWidgets.QCheckBox("Enable move-limit adjudication (Experimental)")
        self.move_limit_guard_checkbox.setChecked(False)
        self.move_limit_guard_checkbox.setToolTip(
            "Non-standard rule.\n"
            "If enabled, game is adjudicated by score when move count reaches board_size*board_size*2."
        )
        self.setup_layout.addRow(self.move_limit_guard_checkbox)

        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItem("Play Black", "black")
        self.side_combo.addItem("Play White", "white")
        self._add_form_row("side", "Your Side", self.side_combo)

        self.agent_combo = QtWidgets.QComboBox()
        self.agent_combo.addItem("MCTS", "mcts")
        self.agent_combo.addItem("Random", "random")
        self.agent_combo.addItem("Minimax", "minimax")
        self.agent_combo.currentIndexChanged.connect(self._sync_agent_options)
        self._add_form_row("ai_agent", self._agent_label_widget("AI Agent"), self.agent_combo)

        self.black_agent_combo = QtWidgets.QComboBox()
        self.black_agent_combo.addItem("MCTS", "mcts")
        self.black_agent_combo.addItem("Random", "random")
        self.black_agent_combo.addItem("Minimax", "minimax")
        self.black_agent_combo.currentIndexChanged.connect(self._sync_agent_options)
        self._add_form_row(
            "black_agent",
            self._agent_label_widget("Black Agent"),
            self.black_agent_combo,
        )

        self.white_agent_combo = QtWidgets.QComboBox()
        self.white_agent_combo.addItem("MCTS", "mcts")
        self.white_agent_combo.addItem("Random", "random")
        self.white_agent_combo.addItem("Minimax", "minimax")
        self.white_agent_combo.currentIndexChanged.connect(self._sync_agent_options)
        self._add_form_row(
            "white_agent",
            self._agent_label_widget("White Agent"),
            self.white_agent_combo,
        )

        self.mcts_rounds = QtWidgets.QSpinBox()
        self.mcts_rounds.setRange(-1, 4000)
        self.mcts_rounds.setValue(-1)
        self.mcts_rounds.setSingleStep(50)
        mcts_rounds_tooltip = (
            "MCTS rounds setting:\n"
            "- -1: auto rounds = max(320, board_size^2 * 24). "
            "For 5x5 this is 600 rounds, and rollouts continue to the end.\n"
            "- Other values: use the selected number of rounds, and cap each rollout at depth 28."
        )
        self.mcts_rounds.setToolTip(mcts_rounds_tooltip)
        self.mcts_rounds.setSpecialValueText("-1 (To End)")
        self._add_form_row(
            "mcts_rounds",
            self._help_label_widget("MCTS Rounds", mcts_rounds_tooltip),
            self.mcts_rounds,
        )

        self.minimax_depth = QtWidgets.QSpinBox()
        self.minimax_depth.setRange(1, 5)
        self.minimax_depth.setValue(3)
        self.minimax_depth.setToolTip("Search depth for Minimax / Alpha-Beta.")
        self._add_form_row("minimax_depth", "Minimax Depth", self.minimax_depth)

        self.new_game_btn = QtWidgets.QPushButton("New Game")
        self.new_game_btn.clicked.connect(self.start_new_game)
        self.setup_layout.addRow(self.new_game_btn)
        return group

    def _build_action_group(self):
        group = QtWidgets.QGroupBox("Actions")
        layout = QtWidgets.QHBoxLayout(group)
        layout.setSpacing(8)

        self.pass_btn = QtWidgets.QPushButton("Pass")
        self.pass_btn.clicked.connect(self.pass_turn)
        layout.addWidget(self.pass_btn)

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_move)
        layout.addWidget(self.undo_btn)

        self.resign_btn = QtWidgets.QPushButton("Resign")
        self.resign_btn.clicked.connect(self.resign_game)
        layout.addWidget(self.resign_btn)
        return group

    def _build_status_group(self):
        group = QtWidgets.QGroupBox("Status")
        layout = QtWidgets.QFormLayout(group)

        self.turn_label = QtWidgets.QLabel("-")
        self.result_label = QtWidgets.QLabel("-")
        self.moves_label = QtWidgets.QLabel("0")
        self.score_label = QtWidgets.QLabel("-")

        layout.addRow("Turn", self.turn_label)
        layout.addRow("Result", self.result_label)
        layout.addRow("Moves", self.moves_label)
        layout.addRow("Score Est.", self.score_label)
        return group

    def _build_log_group(self):
        group = QtWidgets.QGroupBox("Move Log")
        layout = QtWidgets.QVBoxLayout(group)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(600)
        layout.addWidget(self.log_edit)
        return group

    def _add_form_row(self, key, label, field):
        if isinstance(label, str):
            label_widget = QtWidgets.QLabel(label)
        else:
            label_widget = label
        self.setup_layout.addRow(label_widget, field)
        self._form_rows[key] = (label_widget, field)

    def _set_row_visible(self, key, visible):
        label_widget, field = self._form_rows[key]
        label_widget.setVisible(visible)
        field.setVisible(visible)

    def _agent_label_widget(self, title):
        tooltip = (
            "Agent options:\n"
            "- Random: legal random move\n"
            "- MCTS: Monte Carlo Tree Search\n"
            "- Minimax: Alpha-Beta search\n"
            "MCTS rounds = -1 means rollout to terminal state."
        )
        return self._help_label_widget(title, tooltip)

    def _help_label_widget(self, title, tooltip):
        label_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(label_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        text = QtWidgets.QLabel(title)
        help_btn = QtWidgets.QToolButton()
        help_btn.setObjectName("helpButton")
        help_btn.setText("?")
        help_btn.setToolTip(tooltip)
        help_btn.setCursor(QtCore.Qt.WhatsThisCursor)
        help_btn.setAutoRaise(True)

        layout.addWidget(text)
        layout.addWidget(help_btn)
        layout.addStretch(1)
        return label_widget

    def _install_style(self):
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI";
                font-size: 10pt;
                color: #2f2822;
            }
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #faf6ee, stop:0.55 #efe0c8, stop:1 #d9c09c
                );
            }
            QGroupBox {
                border: 1px solid #ad9370;
                border-radius: 12px;
                margin-top: 10px;
                background: rgba(255, 248, 236, 0.82);
                font-weight: 600;
                padding-top: 8px;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 6px;
                color: #4b3a2a;
            }
            QPushButton {
                border: 1px solid #8c6b42;
                border-radius: 9px;
                background: #f6e3c0;
                padding: 7px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f3d7a4;
            }
            QPushButton:pressed {
                background: #e5be82;
            }
            QPushButton:disabled {
                color: #8f826f;
                background: #eadfcf;
                border-color: #b8ab97;
            }
            QComboBox, QSpinBox, QPlainTextEdit {
                border: 1px solid #b99c77;
                border-radius: 8px;
                background: #fffaf1;
                padding: 5px 8px;
            }
            QPlainTextEdit {
                selection-background-color: #d1ab70;
            }
            QToolButton#helpButton {
                border: 1px solid #8c6b42;
                border-radius: 8px;
                background: #f7e5c6;
                color: #5a4327;
                min-width: 16px;
                max-width: 16px;
                min-height: 16px;
                max-height: 16px;
                font-weight: 700;
                padding: 0;
            }
            QToolButton#helpButton:hover {
                background: #f2d49e;
            }
            """
        )

    def _sync_mode_visibility(self):
        human_mode = self._current_mode() == "human_ai"
        self._set_row_visible("side", human_mode)
        self._set_row_visible("ai_agent", human_mode)
        self._set_row_visible("black_agent", not human_mode)
        self._set_row_visible("white_agent", not human_mode)
        self._refresh_controls()

    def _sync_agent_options(self):
        selected_types = self._selected_agent_types()
        show_mcts = "mcts" in selected_types
        show_minimax = "minimax" in selected_types
        self._set_row_visible("mcts_rounds", show_mcts)
        self._set_row_visible("minimax_depth", show_minimax)

    def _selected_agent_types(self):
        if self._current_mode() == "human_ai":
            return [self.agent_combo.currentData() or "random"]
        return [
            self.black_agent_combo.currentData() or "random",
            self.white_agent_combo.currentData() or "random",
        ]

    def _current_mode(self):
        return self.mode_combo.currentData() or "human_ai"

    def _is_game_over(self):
        return self._forced_game_over or self.game_state.is_over()

    def _has_human_player(self):
        return any(kind == "human" for kind in self.player_kinds.values())

    @QtCore.Slot()
    def start_new_game(self):
        if self._busy:
            return

        size = int(self.size_combo.currentText())
        self.game_state = GameState.new_game(size, komi=self._komi_override)
        self.history = [self.game_state]
        self._move_limit = size * size * 2
        self._forced_game_over = False
        self._forced_winner = None
        self._forced_reason = ""
        self._forced_result_text = ""

        self.log_edit.clear()
        mode = self._current_mode()
        if mode == "human_ai":
            side = self.side_combo.currentData() or "black"
            human = Player.black if side == "black" else Player.white
            ai_player = human.other
            ai_type = self.agent_combo.currentData() or "random"
            self.player_kinds = {
                Player.black: "human" if human == Player.black else "ai",
                Player.white: "human" if human == Player.white else "ai",
            }
            self.player_agents = {ai_player: self._build_agent(ai_type)}
            self._append_log(
                f"New game {size}x{size} (komi={self.game_state.komi:.1f}): "
                f"You={self._player_name(human)}, AI={ai_type.upper()}"
            )
        else:
            black_type = self.black_agent_combo.currentData() or "random"
            white_type = self.white_agent_combo.currentData() or "random"
            self.player_kinds = {Player.black: "ai", Player.white: "ai"}
            self.player_agents = {
                Player.black: self._build_agent(black_type),
                Player.white: self._build_agent(white_type),
            }
            self._append_log(
                f"New game {size}x{size} (komi={self.game_state.komi:.1f}): "
                f"Black={black_type.upper()} vs White={white_type.upper()}"
            )

        self.board_widget.set_game_state(self.game_state)
        self._append_log("Game started.")
        if self.move_limit_guard_checkbox.isChecked():
            self._append_log(
                f"[Experimental] Move-limit adjudication is ON (limit={self._move_limit})."
            )
        self._update_status("New game started.")
        self._refresh_controls()

        self._maybe_trigger_next_turn()

    def _build_agent(self, agent_type):
        if agent_type == "mcts":
            rounds = self.mcts_rounds.value()
            rounds = -1 if rounds <= 0 else rounds
            rollout_depth = -1 if rounds == -1 else 28
            return MCTSAgent(num_rounds=rounds, max_rollout_depth=rollout_depth)
        if agent_type == "minimax":
            return MinimaxAgent(max_depth=self.minimax_depth.value())
        return RandomAgent()

    @QtCore.Slot(object)
    def on_human_click(self, point):
        if (
            self._busy
            or self.game_state is None
            or self._is_game_over()
            or self.player_kinds.get(self.game_state.next_player) != "human"
        ):
            return

        move = Move.play(point)
        if not self.game_state.is_valid_move(move):
            self.statusBar().showMessage("Illegal move.")
            return

        self._apply_move(move, actor="You")
        self._maybe_trigger_next_turn()

    @QtCore.Slot()
    def pass_turn(self):
        if (
            self._busy
            or self.game_state is None
            or self._is_game_over()
            or self.player_kinds.get(self.game_state.next_player) != "human"
        ):
            return
        self._apply_move(Move.pass_turn(), actor="You")
        self._maybe_trigger_next_turn()

    @QtCore.Slot()
    def resign_game(self):
        if (
            self._busy
            or self.game_state is None
            or self._is_game_over()
            or self.player_kinds.get(self.game_state.next_player) != "human"
        ):
            return
        self._apply_move(Move.resign(), actor="You")

    @QtCore.Slot()
    def undo_move(self):
        if self._busy or len(self.history) <= 1:
            return

        self._forced_game_over = False
        self._forced_winner = None
        self._forced_reason = ""
        self._forced_result_text = ""

        if self._current_mode() == "human_ai":
            popped = 0
            while len(self.history) > 1:
                self.history.pop()
                popped += 1
                turn = self.history[-1].next_player
                if self.player_kinds.get(turn) == "human":
                    break
        else:
            self.history.pop()
            popped = 1

        self.game_state = self.history[-1]
        self.board_widget.set_game_state(self.game_state)
        self._append_log(f"Undo {popped} ply.")
        self._update_status("Move undone.")
        self._refresh_controls()
        self._maybe_trigger_next_turn()

    def _apply_move(self, move, actor):
        if not self.game_state.is_valid_move(move):
            self.statusBar().showMessage("Illegal move.")
            return False

        self.game_state = self.game_state.apply_move(move)
        self.history.append(self.game_state)
        self.board_widget.set_game_state(self.game_state)
        self._append_log(f"{actor}: {self._format_move(move)}")
        self._check_game_conclusion()
        self._update_status()
        self._refresh_controls()
        return True

    def _check_game_conclusion(self):
        if self.game_state.is_over():
            return
        move_count = len(self.history) - 1
        if self.move_limit_guard_checkbox.isChecked() and move_count >= self._move_limit:
            self._force_finish_by_score(
                f"Move limit {self._move_limit} reached, adjudicated by score."
            )
            return

        if not self._has_play_moves(self.game_state):
            next_state = self.game_state.apply_move(Move.pass_turn())
            if not self._has_play_moves(next_state):
                self._force_finish_by_score(
                    "No legal play move for both players, adjudicated by score."
                )

    def _force_finish_by_score(self, reason):
        result = compute_game_result(self.game_state)
        self._forced_game_over = True
        self._forced_winner = result.winner
        self._forced_reason = reason
        self._forced_result_text = str(result)
        self._append_log(f"Adjudication: {reason} Result={self._forced_result_text}")

    @staticmethod
    def _has_play_moves(game_state):
        return any(move.is_play for move in game_state.legal_moves())

    def _maybe_trigger_next_turn(self):
        if self._busy or self._is_game_over():
            return
        if self.player_kinds.get(self.game_state.next_player) != "ai":
            return
        QtCore.QTimer.singleShot(60, self._trigger_ai_move)

    def _trigger_ai_move(self):
        if self._busy or self._is_game_over():
            return
        if self.player_kinds.get(self.game_state.next_player) != "ai":
            return

        current_player = self.game_state.next_player
        agent = self.player_agents.get(current_player)
        if agent is None:
            return

        self._busy = True
        self._pending_state_key = (
            self.game_state.next_player,
            self.game_state.board.zobrist_hash(),
        )
        self._refresh_controls()
        self.statusBar().showMessage("AI is thinking...")

        self._ai_thread = QtCore.QThread(self)
        self._ai_worker = AIMoveWorker(agent, self.game_state, self._pending_state_key)
        self._ai_worker.moveToThread(self._ai_thread)

        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.finished.connect(self._on_ai_finished)
        self._ai_worker.failed.connect(self._on_ai_failed)
        self._ai_worker.finished.connect(self._ai_thread.quit)
        self._ai_worker.failed.connect(self._ai_thread.quit)
        self._ai_worker.finished.connect(self._ai_worker.deleteLater)
        self._ai_worker.failed.connect(self._ai_worker.deleteLater)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    @QtCore.Slot(object, object)
    def _on_ai_finished(self, move, state_key):
        self._finish_ai_turn()

        current_key = (
            self.game_state.next_player,
            self.game_state.board.zobrist_hash(),
        )
        if state_key != current_key or self._is_game_over():
            return

        acting_player = self.game_state.next_player
        if move is None or not self.game_state.is_valid_move(move):
            move = Move.pass_turn()
        self._apply_move(move, actor=f"{self._player_name(acting_player)} AI")
        self._maybe_trigger_next_turn()

    @QtCore.Slot(str, object)
    def _on_ai_failed(self, message, _state_key):
        self._finish_ai_turn()
        self.statusBar().showMessage(f"AI error: {message}")

    def _finish_ai_turn(self):
        self._busy = False
        self._pending_state_key = None
        self._ai_worker = None
        self._ai_thread = None
        self._refresh_controls()

    def _refresh_controls(self):
        game_over = self.game_state is not None and self._is_game_over()
        has_human = self._has_human_player()
        human_turn = (
            self.game_state is not None
            and self.player_kinds.get(self.game_state.next_player) == "human"
            and not game_over
            and not self._busy
        )

        self.new_game_btn.setEnabled(not self._busy)
        self.undo_btn.setEnabled((len(self.history) > 1) and (not self._busy))
        self.pass_btn.setEnabled(human_turn and has_human)
        self.resign_btn.setEnabled(human_turn and has_human)
        self.board_widget.setEnabled(has_human and human_turn)

    def _update_status(self, message=None):
        self.moves_label.setText(str(max(0, len(self.history) - 1)))
        self.score_label.setText(self._estimate_score_text())

        if self._forced_game_over:
            self.turn_label.setText("Game Over")
            winner_text = "Draw" if self._forced_winner is None else self._player_name(self._forced_winner)
            self.result_label.setText(
                f"Winner: {winner_text} ({self._forced_result_text}) [Adjudicated]"
            )
            self.statusBar().showMessage(self._forced_reason)
            return

        if self.game_state.is_over():
            winner = self.game_state.winner()
            self.turn_label.setText("Game Over")
            self.result_label.setText(
                "Winner: " + ("Draw" if winner is None else self._player_name(winner))
            )
            self.statusBar().showMessage("Game over.")
            return

        self.turn_label.setText(self._player_name(self.game_state.next_player))
        self.result_label.setText("In Progress")
        self.statusBar().showMessage(message or "Ready")

    def _estimate_score_text(self):
        territory = evaluate_territory(self.game_state.board)
        black = territory.num_black_stones + territory.num_black_territory
        white = territory.num_white_stones + territory.num_white_territory + self.game_state.komi
        return f"B {black:.1f} - W {white:.1f}"

    def _player_name(self, player):
        return "Black" if player == Player.black else "White"

    def _format_move(self, move):
        if move.is_pass:
            return "pass"
        if move.is_resign:
            return "resign"
        return f"({move.point.row}, {move.point.col})"

    def _append_log(self, text):
        self.log_edit.appendPlainText(text)
        bar = self.log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())
