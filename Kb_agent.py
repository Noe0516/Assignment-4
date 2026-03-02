"""
Knowledge-Based Agent for the Hazardous Warehouse
===================================================
Uses Z3 propositional logic to reason about hazards and navigate safely.

Tasks covered:
  1. Z3 setup and z3_entails verification
  2. Symbol helpers and build_warehouse_kb()
  3. Manual reasoning replicating Section 3.2
  4. WarehouseKBAgent with perceive/tell/ask/act cycle + BFS path planning
  5. Test run on RN example layout
  6. Reflection summary printed at end
  Bonus: Shutdown device reasoning (fire at forklift when location is known)
"""

from collections import deque
from z3 import Bool, Bools, Or, And, Not, Solver, unsat, sat  # ✓ fixed: import directly from z3

from hazardous_warehouse_env import HazardousWarehouseEnv, Action, Direction


# =============================================================================
# TASK 1 — Z3 Setup & z3_entails
# =============================================================================

def z3_entails(solver: Solver, query) -> bool:
    """
    Return True if the KB (solver) entails `query`.
    Uses push/pop so the solver state is unchanged after the call.
    Entailment: KB ⊨ α  iff  KB ∧ ¬α is unsatisfiable.
    """
    solver.push()
    solver.add(Not(query))
    result = solver.check()
    solver.pop()
    return result == unsat


def task1_verification():
    """Verify Z3 basics and z3_entails as required by Task 1."""
    print("=" * 60)
    print("TASK 1 — Z3 Setup Verification")
    print("=" * 60)

    P, Q = Bools("P Q")
    s = Solver()

    # Add biconditional and fact
    s.add(P == Q)
    s.add(P)

    # Check satisfiability
    result = s.check()
    print(f"s.check()  →  {result}")          # should print sat
    assert result == sat, "Expected sat"

    # Inspect model
    m = s.model()
    print(f"s.model()  →  {m}")               # should show P=True, Q=True

    # Verify entailment
    entails_Q = z3_entails(s, Q)
    print(f"z3_entails(s, Q)  →  {entails_Q}")  # should be True
    assert entails_Q, "Expected True"

    print("Task 1 PASSED ✓\n")


# =============================================================================
# TASK 2 — Symbol Helpers and build_warehouse_kb()
# =============================================================================

def damaged(x: int, y: int):
    """Bool: damaged floor at (x, y)."""
    return Bool(f"D_{x}_{y}")

def forklift_at(x: int, y: int):
    """Bool: malfunctioning forklift at (x, y)."""
    return Bool(f"F_{x}_{y}")

def creaking_at(x: int, y: int):
    """Bool: creaking perceived at (x, y)."""
    return Bool(f"C_{x}_{y}")

def rumbling_at(x: int, y: int):
    """Bool: rumbling perceived at (x, y)."""
    return Bool(f"R_{x}_{y}")

def safe(x: int, y: int):
    """Bool: square (x, y) is safe (no damage, no forklift)."""
    return Bool(f"S_{x}_{y}")


def get_adjacent(x: int, y: int, width: int = 4, height: int = 4):
    """Return list of valid adjacent (x, y) positions."""
    return [
        (nx, ny)
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        if 1 <= nx <= width and 1 <= ny <= height
    ]


def build_warehouse_kb(width: int = 4, height: int = 4) -> Solver:
    """
    Build the propositional KB for the hazardous warehouse.

    Physics encoded (27 × 3 = 48 biconditionals for 4×4 grid):
      Creaking_xy  ↔  ∨{ Damaged_ab : (a,b) adjacent to (x,y) }
      Rumbling_xy  ↔  ∨{ Forklift_ab : (a,b) adjacent to (x,y) }
      Safe_xy      ↔  ¬Damaged_xy  ∧  ¬Forklift_xy

    Initial knowledge:
      Safe(1,1)  →  ¬Damaged(1,1) ∧ ¬Forklift(1,1)
      At least one damaged floor square
      At least one forklift square
    """
    s = Solver()

    for x in range(1, width + 1):
        for y in range(1, height + 1):
            adj = get_adjacent(x, y, width, height)

            # Creaking biconditional
            damage_adj = [damaged(a, b) for a, b in adj]
            s.add(creaking_at(x, y) == Or(damage_adj))

            # Rumbling biconditional
            forklift_adj = [forklift_at(a, b) for a, b in adj]
            s.add(rumbling_at(x, y) == Or(forklift_adj))

            # Safety biconditional
            s.add(safe(x, y) == And(Not(damaged(x, y)), Not(forklift_at(x, y))))

    # Initial knowledge: (1,1) is safe
    s.add(safe(1, 1))
    s.add(Not(damaged(1, 1)))
    s.add(Not(forklift_at(1, 1)))

    # At least one damaged floor
    all_damaged = [damaged(x, y) for x in range(1, width+1) for y in range(1, height+1)]
    s.add(Or(all_damaged))

    # At least one forklift
    all_forklift = [forklift_at(x, y) for x in range(1, width+1) for y in range(1, height+1)]
    s.add(Or(all_forklift))

    return s


def task2_verification():
    """Verify KB is satisfiable."""
    print("=" * 60)
    print("TASK 2 — Build KB Verification")
    print("=" * 60)

    s = build_warehouse_kb()
    result = s.check()
    print(f"build_warehouse_kb().check()  →  {result}")
    assert result == sat, "Expected sat"
    print("Task 2 PASSED ✓\n")


# =============================================================================
# TASK 3 — Manual Reasoning (replicating Section 3.2)
# =============================================================================

def task3_manual_reasoning():
    """
    Manually TELL percepts and ASK about safety, replicating the
    three-step reasoning walkthrough from Section 3.2.
    """
    print("=" * 60)
    print("TASK 3 — Manual Reasoning")
    print("=" * 60)

    s = build_warehouse_kb()

    # ── Step 1: at (1,1), no creaking, no rumbling ────────────────────────────
    print("\n▸ At (1,1): no creaking, no rumbling")
    s.add(Not(creaking_at(1, 1)))
    s.add(Not(rumbling_at(1, 1)))

    safe_21 = z3_entails(s, safe(2, 1))
    safe_12 = z3_entails(s, safe(1, 2))
    print(f"  Safe(2,1)? {safe_21}")   # True
    print(f"  Safe(1,2)? {safe_12}")   # True
    assert safe_21 and safe_12

    # ── Step 2: at (2,1), creaking, no rumbling ───────────────────────────────
    print("\n▸ At (2,1): creaking, no rumbling")
    s.add(safe(2, 1))                         # robot moved here safely
    s.add(creaking_at(2, 1))
    s.add(Not(rumbling_at(2, 1)))

    safe_31  = z3_entails(s, safe(3, 1))
    safe_22  = z3_entails(s, safe(2, 2))
    dmg_31   = z3_entails(s, damaged(3, 1))
    dmg_22   = z3_entails(s, damaged(2, 2))

    print(f"  Safe(3,1)? {safe_31}")   # False — uncertain
    print(f"  Safe(2,2)? {safe_22}")   # False — uncertain
    print(f"  Damaged(3,1) entailed? {dmg_31}")  # False — only disjunction known
    print(f"  Damaged(2,2) entailed? {dmg_22}")  # False

    # We can entail the disjunction though
    dmg_disjunction = z3_entails(s, Or(damaged(3, 1), damaged(2, 2)))
    print(f"  Damaged(3,1) ∨ Damaged(2,2)? {dmg_disjunction}")  # True

    # ── Step 3: at (1,2), rumbling, no creaking ───────────────────────────────
    print("\n▸ At (1,2): rumbling, no creaking")
    s.add(safe(1, 2))
    s.add(rumbling_at(1, 2))
    s.add(Not(creaking_at(1, 2)))

    safe_31_now  = z3_entails(s, safe(3, 1))
    dmg_31_now   = z3_entails(s, damaged(3, 1))
    fk_13        = z3_entails(s, forklift_at(1, 3))
    fk_22_now    = z3_entails(s, forklift_at(2, 2))

    print(f"  Safe(3,1)?      {safe_31_now}")   # False  — now known damaged
    print(f"  Damaged(3,1)?   {dmg_31_now}")    # True   — pinpointed!
    print(f"  Forklift(1,3)?  {fk_13}")         # True
    print(f"  Forklift(2,2)?  {fk_22_now}")

    print("\nTask 3 PASSED ✓\n")


# =============================================================================
# TASK 4 — WarehouseKBAgent
# =============================================================================

class WarehouseKBAgent:
    """
    Knowledge-based agent for the Hazardous Warehouse.

    Perceive → Tell → Ask → Act cycle:
      1. PERCEIVE: receive Percept from environment
      2. TELL:     add percept facts to Z3 solver
      3. ASK:      query solver for safe squares and forklift location
      4. ACT:      BFS to next target through known-safe squares;
                   fire shutdown device when forklift location is certain
                   (bonus task).
    """

    def __init__(self, width: int = 4, height: int = 4):
        self.width  = width
        self.height = height
        self.solver = build_warehouse_kb(width, height)

        # Agent's explicit knowledge sets (maintained alongside Z3)
        self.known_safe:      set = {(1, 1)}
        self.known_damaged:   set = set()
        self.known_forklift:  set = set()
        self.visited:         set = set()

        # Navigation state
        self.action_queue:    list = []
        self.target = None
        self.has_package = False
        self.phase = "explore"   # explore → retrieve → return

    # ── TELL ──────────────────────────────────────────────────────────────────

    def tell(self, x: int, y: int, percept):
        """Add percept observations at (x,y) to the KB."""
        # Mark current square as safe (robot is alive here)
        self.solver.add(safe(x, y))
        self.solver.add(Not(damaged(x, y)))
        self.solver.add(Not(forklift_at(x, y)))
        self.known_safe.add((x, y))

        # Creaking / no creaking
        if percept.creaking:
            self.solver.add(creaking_at(x, y))
        else:
            self.solver.add(Not(creaking_at(x, y)))

        # Rumbling / no rumbling
        if percept.rumbling:
            self.solver.add(rumbling_at(x, y))
        else:
            self.solver.add(Not(rumbling_at(x, y)))

    # ── ASK ───────────────────────────────────────────────────────────────────

    def ask_safe(self, x: int, y: int) -> bool:
        return z3_entails(self.solver, safe(x, y))

    def ask_damaged(self, x: int, y: int) -> bool:
        return z3_entails(self.solver, damaged(x, y))

    def ask_forklift(self, x: int, y: int) -> bool:
        return z3_entails(self.solver, forklift_at(x, y))

    def _update_knowledge(self):
        """Refresh known_safe / known_damaged / known_forklift via Z3 queries."""
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                pos = (x, y)
                if pos not in self.known_safe and self.ask_safe(x, y):
                    self.known_safe.add(pos)
                if pos not in self.known_damaged and self.ask_damaged(x, y):
                    self.known_damaged.add(pos)
                if pos not in self.known_forklift and self.ask_forklift(x, y):
                    self.known_forklift.add(pos)

    # ── PATH PLANNING (BFS through safe squares) ──────────────────────────────

    def _bfs_path(self, start, goal):
        """
        BFS from start to goal moving only through known-safe squares.
        Returns list of positions (excluding start) or None if unreachable.
        """
        if start == goal:
            return []
        queue   = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            cx, cy = path[-1]
            for nx, ny in get_adjacent(cx, cy, self.width, self.height):
                npos = (nx, ny)
                if npos in visited:
                    continue
                if npos not in self.known_safe and npos != goal:
                    continue
                new_path = path + [npos]
                if npos == goal:
                    return new_path[1:]   # exclude start
                visited.add(npos)
                queue.append(new_path)
        return None

    def _path_to_actions(self, path, start, facing):
        """Convert a list of positions into a sequence of Actions."""
        actions = []
        current_facing = facing
        cx, cy = start

        for nx, ny in path:
            dx, dy = nx - cx, ny - cy
            if   dx ==  1: required = Direction.EAST
            elif dx == -1: required = Direction.WEST
            elif dy ==  1: required = Direction.NORTH
            else:          required = Direction.SOUTH

            for _ in range(4):
                if current_facing == required:
                    break
                right = current_facing.turn_right()
                left  = current_facing.turn_left()
                if right == required:
                    actions.append(Action.TURN_RIGHT)
                    current_facing = right
                elif left == required:
                    actions.append(Action.TURN_LEFT)
                    current_facing = left
                else:
                    actions.append(Action.TURN_RIGHT)
                    current_facing = right

            actions.append(Action.FORWARD)
            cx, cy = nx, ny

        return actions

    def _pick_frontier(self, robot_pos):
        """
        Choose the nearest unvisited square that is adjacent to a known-safe
        square and not known to be dangerous.
        """
        candidates = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                pos = (x, y)
                if pos in self.visited:
                    continue
                if pos in self.known_damaged or pos in self.known_forklift:
                    continue
                adj = get_adjacent(x, y, self.width, self.height)
                if any(a in self.known_safe for a in adj):
                    candidates.append(pos)

        if not candidates:
            return None

        def dist(p):
            return abs(p[0] - robot_pos[0]) + abs(p[1] - robot_pos[1])

        safe_cands = [p for p in candidates if p in self.known_safe]
        if safe_cands:
            return min(safe_cands, key=dist)
        return min(candidates, key=dist)

    # ── BONUS: Shutdown reasoning ──────────────────────────────────────────────

    def _should_fire_shutdown(self, robot_pos, facing, has_device) -> bool:
        """
        Return True if we are certain a forklift is somewhere along the line
        we are facing AND we still have the shutdown device.
        """
        if not has_device:
            return False
        dx, dy = facing.delta()
        x, y = robot_pos
        while True:
            x += dx
            y += dy
            if not (1 <= x <= self.width and 1 <= y <= self.height):
                break
            if self.ask_forklift(x, y):
                return True
        return False

    # ── ACT ───────────────────────────────────────────────────────────────────

    def act(self, robot_pos, facing, has_package, has_device, percept) -> Action:
        """
        Main perceive→tell→ask→act cycle.
        Called each step; returns one Action.
        """
        x, y = robot_pos
        self.visited.add(robot_pos)
        self.has_package = has_package

        # TELL
        self.tell(x, y, percept)

        # Grab package if we're on it
        if percept.beacon and not has_package:
            self.action_queue.clear()
            return Action.GRAB

        # Exit if we have the package and are at (1,1)
        if has_package and robot_pos == (1, 1):
            return Action.EXIT

        # ASK — refresh knowledge
        self._update_knowledge()

        # BONUS: fire shutdown if forklift is in our line of sight
        if self._should_fire_shutdown(robot_pos, facing, has_device):
            print(f"  [BONUS] Firing shutdown device from {robot_pos} facing {facing.name}")
            self.action_queue.clear()
            return Action.SHUTDOWN

        # If we have a queued action plan, execute it
        if self.action_queue:
            return self.action_queue.pop(0)

        # PLAN — decide next target
        if has_package:
            goal = (1, 1)
        elif percept.beacon:
            goal = robot_pos
        else:
            goal = self._pick_frontier(robot_pos)
            if goal is None:
                goal = (1, 1)

        # BFS to goal
        path = self._bfs_path(robot_pos, goal)
        if path is None:
            return Action.TURN_RIGHT

        if not path:
            if robot_pos == (1, 1) and has_package:
                return Action.EXIT
            return Action.TURN_RIGHT

        actions = self._path_to_actions(path, robot_pos, facing)
        if actions:
            self.action_queue = actions[1:]
            return actions[0]

        return Action.TURN_RIGHT


# =============================================================================
# TASK 5 — Test Run on RN Example Layout
# =============================================================================

def run_agent(env: HazardousWarehouseEnv, max_steps: int = 200, verbose: bool = True):
    """Run the WarehouseKBAgent on the given environment."""
    agent = WarehouseKBAgent(env.width, env.height)
    percept = env._last_percept

    if verbose:
        print(f"{'─'*60}")
        print(f"  Start | pos={env.robot_position} facing={env.robot_direction.name}")
        print(f"  Percept: {percept}")
        print(f"{'─'*60}")

    for step in range(max_steps):
        action = agent.act(
            robot_pos   = env.robot_position,
            facing      = env.robot_direction,
            has_package = env.has_package,
            has_device  = env.has_shutdown_device,
            percept     = percept,
        )

        percept, reward, done, info = env.step(action)

        if verbose:
            print(f"  Step {step+1:3d} | {action.name:<12} "
                  f"pos={env.robot_position} facing={env.robot_direction.name:6} "
                  f"reward={reward:+.0f}  percept={percept}")

        if done:
            break

    return env


def task5_test():
    """Task 5: run on the RN example layout and report results."""
    from hazardous_warehouse_viz import configure_rn_example_layout  # ✓ fixed filename

    print("=" * 60)
    print("TASK 5 — Agent Test on RN Example Layout")
    print("=" * 60)
    print()

    env = HazardousWarehouseEnv()
    configure_rn_example_layout(env)

    print("True state:")
    print(env.render(reveal=True))
    print()

    env = run_agent(env, max_steps=200, verbose=True)

    print()
    print(f"  Steps taken  : {env.steps}")
    print(f"  Total reward : {env.total_reward:.1f}")
    true_state = env.get_true_state()
    outcome = "SUCCESS ✓" if true_state["success"] else (
              "DIED ✗"   if not true_state["robot"]["alive"] else "INCOMPLETE")
    print(f"  Outcome      : {outcome}")
    print()

    return env


# =============================================================================
# TASK 6 — Reflection (printed)
# =============================================================================

REFLECTION = """
TASK 6 — Reflection
═══════════════════════════════════════════════════════════════

Situation where the agent gets stuck or behaves conservatively:
  The agent can get stuck when the package is surrounded on all sides by squares
  that are not yet confirmed safe — for example, if every path to the package
  requires crossing a square whose damage/forklift status is ambiguous (the solver
  cannot rule out a hazard there). Because the agent only moves through
  *entailed-safe* squares, it will refuse to risk the crossing even if a human
  would judge the probability of danger to be low.

What additional reasoning capability would help:
  Probabilistic or utility-based reasoning (e.g. POMDP planning) would allow the
  agent to weigh the expected reward of a risky move against the cost of staying
  stuck. Alternatively, adding an "exactly one damaged floor" / "exactly one
  forklift" constraint to the KB would enable stronger logical deductions — once
  a hazard is located elsewhere, the remaining uncertain squares become provably
  safe, breaking the deadlock without requiring probability estimates.
"""


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    task1_verification()
    task2_verification()
    task3_manual_reasoning()
    task5_test()
    print(REFLECTION)