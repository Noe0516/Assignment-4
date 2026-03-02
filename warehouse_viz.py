"""
Hazardous Warehouse Visualization

Visualization and animation utilities for the Hazardous Warehouse environment.
Provides grid rendering, percept display, and episode replay animations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazardous_warehouse_env import HazardousWarehouseEnv


# -----------------------------------------------------------------------------
# Color Schemes
# -----------------------------------------------------------------------------

COLORS = {
    "empty": (0.95, 0.95, 0.95),
    "wall": (0.2, 0.2, 0.2),
    "unknown": (0.7, 0.7, 0.7),
    "exit": (0.3, 0.8, 0.3),
    "damaged": (1.0, 0.5, 0.0),
    "forklift": (1.0, 0.0, 0.0),
    "forklift_dead": (0.5, 0.5, 0.5),
    "package": (1.0, 0.85, 0.0),
    "robot": (0.2, 0.5, 0.9),
    "robot_loaded": (0.6, 0.2, 0.8),
    "robot_dead": (0.1, 0.1, 0.1),
    "creaking": (1.0, 0.5, 0.0),
    "rumbling": (1.0, 0.0, 0.0),
    "safe": (0.7, 0.95, 0.7),
    "uncertain": (1.0, 0.8, 0.5),
}


# -----------------------------------------------------------------------------
# Grid Rendering
# -----------------------------------------------------------------------------

def state_to_grid(
    env: "HazardousWarehouseEnv",
    reveal: bool = False,
    show_percepts: bool = True,
    known_safe=None,
    known_dangerous=None,
):
    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]

    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"]) if true_state["forklift"] else None
    forklift_alive = true_state["forklift_alive"]
    package = tuple(true_state["package"]) if true_state["package"] else None
    robot = true_state["robot"]
    robot_pos = (robot["x"], robot["y"])

    known_safe = known_safe or set()
    known_dangerous = known_dangerous or set()

    grid = []

    for row in range(height, 0, -1):
        grid_row = []
        for col in range(1, width + 1):
            pos = (col, row)

            if pos == robot_pos:
                if not robot["alive"]:
                    color = COLORS["robot_dead"]
                elif robot["has_package"]:
                    color = COLORS["robot_loaded"]
                else:
                    color = COLORS["robot"]
            elif reveal:
                if pos in damaged:
                    color = COLORS["damaged"]
                elif forklift and pos == forklift:
                    color = COLORS["forklift"] if forklift_alive else COLORS["forklift_dead"]
                elif package and pos == package and not robot["has_package"]:
                    color = COLORS["package"]
                elif pos == (1, 1):
                    color = COLORS["exit"]
                else:
                    color = COLORS["empty"]
            else:
                if pos in known_dangerous:
                    color = COLORS["damaged"]
                elif pos in known_safe:
                    color = COLORS["exit"] if pos == (1, 1) else COLORS["safe"]
                else:
                    color = COLORS["unknown"]

            grid_row.append(color)

        grid.append(grid_row)

    return grid


# -----------------------------------------------------------------------------
# Static Plot
# -----------------------------------------------------------------------------

def plot_state(
    env: "HazardousWarehouseEnv",
    ax=None,
    reveal=False,
    known_safe=None,
    known_dangerous=None,
    title=None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    grid = state_to_grid(
        env,
        reveal=reveal,
        known_safe=known_safe,
        known_dangerous=known_dangerous,
    )

    ax.imshow(grid, interpolation="nearest", aspect="equal")

    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    robot = true_state["robot"]

    for i in range(width + 1):
        ax.axvline(i - 0.5, color="gray", linewidth=0.5)
    for i in range(height + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5)

    ax.set_xticks(range(width))
    ax.set_xticklabels(range(1, width + 1))
    ax.set_yticks(range(height))
    ax.set_yticklabels(range(height, 0, -1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Step {env.steps}")

    return ax


# -----------------------------------------------------------------------------
# Replay Animation
# -----------------------------------------------------------------------------

def replay_episode(history, env, interval_ms=500, reveal=True):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("matplotlib not available")
        return

    if not history:
        print("No history to replay")
        return

    true_state = env.get_true_state()
    width, height = true_state["width"], true_state["height"]
    damaged = set(tuple(p) for p in true_state["damaged"])
    forklift = tuple(true_state["forklift"]) if true_state["forklift"] else None
    package = tuple(true_state["package"]) if true_state["package"] else None

    fig, ax = plt.subplots(figsize=(6, 6))

    def build_grid(state):
        grid = []
        for row in range(height, 0, -1):
            grid_row = []
            for col in range(1, width + 1):
                pos = (col, row)
                robot_pos = (state["robot_x"], state["robot_y"])

                if pos == robot_pos:
                    if not state["alive"]:
                        color = COLORS["robot_dead"]
                    elif state["has_package"]:
                        color = COLORS["robot_loaded"]
                    else:
                        color = COLORS["robot"]
                elif reveal:
                    if pos in damaged:
                        color = COLORS["damaged"]
                    elif forklift and pos == forklift:
                        color = COLORS["forklift"]
                    elif package and pos == package:
                        color = COLORS["package"]
                    elif pos == (1, 1):
                        color = COLORS["exit"]
                    else:
                        color = COLORS["empty"]
                else:
                    color = COLORS["unknown"]

                grid_row.append(color)
            grid.append(grid_row)
        return grid

    im = ax.imshow(build_grid(history[0]), interpolation="nearest", aspect="equal")

    def update(frame_idx):
        state = history[frame_idx]
        im.set_data(build_grid(state))
        ax.set_title(f"Step {state['step']} | Action: {state['action']}")
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=interval_ms, blit=False
    )

    plt.show()
    return anim


# -----------------------------------------------------------------------------
# Example Run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from hazardous_warehouse_env import HazardousWarehouseEnv, Action

    env = HazardousWarehouseEnv(seed=42)

    actions = [
        Action.FORWARD,
        Action.FORWARD,
        Action.TURN_LEFT,
        Action.FORWARD,
    ]

    for action in actions:
        percept, reward, done, info = env.step(action)
        if done:
            break

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_state(env, ax=axes[0], reveal=False, title="Agent View")
        plot_state(env, ax=axes[1], reveal=True, title="True State")
        plt.tight_layout()
        plt.show()

        print("Replaying episode...")
        replay_episode(env.history, env)

    except ImportError:
        print("matplotlib not available")