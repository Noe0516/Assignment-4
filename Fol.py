"""
FOL Knowledge-Based Agent for the Hazardous Warehouse
Section 3.7 – First-Order Logic Agent with Z3
"""

from z3 import (
    DeclareSort, Function, BoolSort, Const,
    ForAll, Exists, Distinct,
    Or, And, Not, Solver, unsat
)

from hazardous_warehouse_env import (
    HazardousWarehouseEnv,
    Action,
)
from warehouse_viz import configure_rn_example_layout


# ============================================================
# Z3 ENTAILMENT
# ============================================================

def z3_entails(solver, query):
    solver.push()
    solver.add(Not(query))
    result = solver.check() == unsat
    solver.pop()
    return result


# ============================================================
# BUILD FOL KNOWLEDGE BASE
# ============================================================

def build_warehouse_kb_fol(width=4, height=4):

    s = Solver()

    # ----- Sort -----
    Location = DeclareSort("Location")

    # ----- Predicates -----
    Damaged  = Function("Damaged", Location, BoolSort())
    Forklift = Function("Forklift", Location, BoolSort())
    Safe     = Function("Safe", Location, BoolSort())
    Creaking = Function("Creaking", Location, BoolSort())
    Rumbling = Function("Rumbling", Location, BoolSort())
    Adjacent = Function("Adjacent", Location, Location, BoolSort())

    # ----- Location constants -----
    loc = {}
    for x in range(1, width+1):
        for y in range(1, height+1):
            loc[(x, y)] = Const(f"L_{x}_{y}", Location)

    # ----- Domain closure -----
    L = Const("L", Location)

    s.add(
        ForAll(
            L,
            Or([L == loc[(x, y)]
                for x in range(1, width+1)
                for y in range(1, height+1)])
        )
    )

    s.add(
        Distinct([loc[(x, y)]
                  for x in range(1, width+1)
                  for y in range(1, height+1)])
    )

    # ----- Adjacency -----
    def get_adjacent(x, y):
        candidates = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        return [(a,b) for a,b in candidates
                if 1 <= a <= width and 1 <= b <= height]

    adj_pairs = []
    for x in range(1, width+1):
        for y in range(1, height+1):
            for (a,b) in get_adjacent(x,y):
                adj_pairs.append((loc[(x,y)], loc[(a,b)]))

    L1 = Const("L1", Location)
    L2 = Const("L2", Location)

    s.add(
        ForAll(
            [L1, L2],
            Adjacent(L1, L2) ==
            Or([And(L1 == p[0], L2 == p[1]) for p in adj_pairs])
        )
    )

    # ----- Physics rules -----
    Lp = Const("Lp", Location)

    s.add(
        ForAll(
            L,
            Creaking(L) ==
            Exists(Lp, And(Adjacent(L, Lp), Damaged(Lp)))
        )
    )

    s.add(
        ForAll(
            L,
            Rumbling(L) ==
            Exists(Lp, And(Adjacent(L, Lp), Forklift(Lp)))
        )
    )

    s.add(
        ForAll(
            L,
            Safe(L) ==
            And(Not(Damaged(L)), Not(Forklift(L)))
        )
    )

    # Initial square safe
    s.add(Not(Damaged(loc[(1,1)])))
    s.add(Not(Forklift(loc[(1,1)])))

    preds = {
        "Damaged": Damaged,
        "Forklift": Forklift,
        "Safe": Safe,
        "Creaking": Creaking,
        "Rumbling": Rumbling
    }

    return s, loc, preds


# ============================================================
# MANUAL REASONING
# ============================================================

def manual_reasoning_fol():

    print("="*60)
    print("FOL Manual Reasoning")
    print("="*60)

    s, loc, P = build_warehouse_kb_fol()

    s.add(Not(P["Creaking"](loc[(1,1)])))
    s.add(Not(P["Rumbling"](loc[(1,1)])))

    print("Safe(2,1)?", z3_entails(s, P["Safe"](loc[(2,1)])))
    print("Safe(1,2)?", z3_entails(s, P["Safe"](loc[(1,2)])))

    s.add(P["Creaking"](loc[(2,1)]))
    s.add(Not(P["Rumbling"](loc[(2,1)])))

    print("Safe(3,1)?", z3_entails(s, P["Safe"](loc[(3,1)])))
    print("Safe(2,2)?", z3_entails(s, P["Safe"](loc[(2,2)])))

    s.add(Not(P["Creaking"](loc[(1,2)])))
    s.add(P["Rumbling"](loc[(1,2)]))

    print("Damaged(3,1)?", z3_entails(s, P["Damaged"](loc[(3,1)])))
    print("Forklift(1,3)?", z3_entails(s, P["Forklift"](loc[(1,3)])))

    print()


# ============================================================
# RUN AGENT
# ============================================================

# ============================================================
# RUN AGENT
# ============================================================

def run_agent():

    env = HazardousWarehouseEnv(seed=0)
    configure_rn_example_layout(env)

    print("True state (hidden from agent):")
    print(env.render(reveal=True))
    print()

    solver, loc, preds = build_warehouse_kb_fol(env.width, env.height)

    visited = set()
    percept = env._last_percept

    for step in range(200):

        current_pos = env.robot_position
        visited.add(current_pos)

        L = loc[current_pos]

        # Add percepts
        solver.add(preds["Creaking"](L) if percept.creaking else Not(preds["Creaking"](L)))
        solver.add(preds["Rumbling"](L) if percept.rumbling else Not(preds["Rumbling"](L)))

        # Find neighbors
        x, y = current_pos
        neighbors = [(nx, ny)
                     for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                     if 1 <= nx <= env.width and 1 <= ny <= env.height]

        safe_moves = [n for n in neighbors
                      if z3_entails(solver, preds["Safe"](loc[n]))]

        next_pos = None
        for n in safe_moves:
            if n not in visited:
                next_pos = n
                break

        if next_pos is None and safe_moves:
            next_pos = safe_moves[0]

        if next_pos is None and neighbors:
            next_pos = neighbors[0]

        if next_pos is None:
            print("No moves available.")
            break

        cx, cy = current_pos
        nx, ny = next_pos

        # Determine relative direction needed
        dx = nx - cx
        dy = ny - cy

        current_dir = env.robot_direction

        # Map direction enum to movement vector
        direction_vectors = {
            current_dir.NORTH: (0, 1),
            current_dir.SOUTH: (0, -1),
            current_dir.EAST:  (1, 0),
            current_dir.WEST:  (-1, 0),
        }

        # Current facing vector
        facing = direction_vectors[current_dir]

        desired = (dx, dy)

        # Left and right rotation maps
        left_turn = {
            current_dir.NORTH: current_dir.WEST,
            current_dir.WEST:  current_dir.SOUTH,
            current_dir.SOUTH: current_dir.EAST,
            current_dir.EAST:  current_dir.NORTH,
        }

        right_turn = {
            current_dir.NORTH: current_dir.EAST,
            current_dir.EAST:  current_dir.SOUTH,
            current_dir.SOUTH: current_dir.WEST,
            current_dir.WEST:  current_dir.NORTH,
        }

        if desired == facing:
            action = Action.FORWARD
        elif direction_vectors[left_turn[current_dir]] == desired:
            action = Action.TURN_LEFT
        elif direction_vectors[right_turn[current_dir]] == desired:
            action = Action.TURN_RIGHT
        else:
            action = Action.TURN_LEFT  # turn around fallback

        percept, reward, done, info = env.step(action)

        print(action.name, env.robot_position, reward)

        if done:
            break

    print("Steps:", env.steps)
    print("Total reward:", env.total_reward)


# ============================================================
# REFLECTION
# ============================================================

REFLECTION = """
Reflection:

The propositional encoding grows quickly because every square needs its
own Boolean variables and physics rules, so the number of rules increases
with grid size. The FOL encoding is more compact because each physics rule
is written once using quantifiers and applies to all locations.

FOL is more readable because the quantified rules closely match their
English descriptions. Domain closure is necessary because without it,
Z3 can invent extra phantom locations to satisfy existential statements.
The propositional encoding does not need domain closure because its
universe is already fixed by explicitly declared Boolean variables.
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    manual_reasoning_fol()
    run_agent()
    print(REFLECTION)