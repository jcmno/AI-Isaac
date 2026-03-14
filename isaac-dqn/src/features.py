from math import sqrt

import numpy as np

from .protocol import GameState


def _nearest_enemies(state: GameState, k: int = 2):
    if not state.enemies:
        return []
    ordered = sorted(
        state.enemies,
        key=lambda e: (e.x - state.player_x) ** 2 + (e.y - state.player_y) ** 2,
    )
    return ordered[:k]


def _nearest_projectiles(state: GameState, k: int = 2):
    if not state.projectiles:
        return []
    ordered = sorted(
        state.projectiles,
        key=lambda t: (t.x - state.player_x) ** 2 + (t.y - state.player_y) ** 2,
    )
    return ordered[:k]


def _in_corner_zone(player_grid: int | None, grid_w: int, grid_size: int, band: int = 4) -> float:
    if player_grid is None or grid_w <= 0 or grid_size <= 0:
        return 0.0
    x = player_grid % grid_w
    y = player_grid // grid_w
    h = max(1, grid_size // grid_w)
    left = x <= band
    right = x >= grid_w - 1 - band
    top = y <= band
    bottom = y >= h - 1 - band
    return 1.0 if ((left and top) or (left and bottom) or (right and top) or (right and bottom)) else 0.0


def _line_blocked_between_grids(src_grid: int | None, dst_grid: int | None, grid_w: int, blocked: set[int]) -> bool:
    """Check if a same-row/column line between src and dst is blocked."""
    if src_grid is None or dst_grid is None or grid_w <= 0:
        return False

    sx, sy = src_grid % grid_w, src_grid // grid_w
    dx, dy = dst_grid % grid_w, dst_grid // grid_w

    if sx == dx:
        step = grid_w if dy > sy else -grid_w
        cur = src_grid + step
        while cur != dst_grid:
            if cur in blocked:
                return True
            cur += step
        return False

    if sy == dy:
        step = 1 if dx > sx else -1
        cur = src_grid + step
        while cur != dst_grid:
            if cur in blocked:
                return True
            cur += step
        return False

    return False


def _incoming_projectile_threat(state: GameState, proj) -> float:
    if proj is None:
        return 0.0
    rx = proj.x - state.player_x
    ry = proj.y - state.player_y
    dist = sqrt(rx * rx + ry * ry)
    if dist < 1e-6:
        return 1.0

    speed = sqrt(proj.vx * proj.vx + proj.vy * proj.vy)
    if speed < 1e-6:
        return 0.0

    # Positive when velocity points toward player.
    toward = max(0.0, -((rx * proj.vx + ry * proj.vy) / (dist * speed + 1e-6)))
    dist_term = max(0.0, 1.0 - min(dist / 240.0, 1.0))
    speed_term = min(speed / 12.0, 1.0)
    return min(toward * dist_term * speed_term, 1.0)


def _mobility_4(state: GameState) -> tuple[float, float, float, float]:
    if state.player_grid is None or state.grid_w <= 0 or state.grid_size <= 0:
        return 0.0, 0.0, 0.0, 0.0

    blocked = state.blocked | state.hazards
    idx = state.player_grid
    x = idx % state.grid_w
    y = idx // state.grid_w
    h = max(1, state.grid_size // state.grid_w)

    up_idx = idx - state.grid_w if y > 0 else None
    down_idx = idx + state.grid_w if y < h - 1 else None
    left_idx = idx - 1 if x > 0 else None
    right_idx = idx + 1 if x < state.grid_w - 1 else None

    can_up = 1.0 if up_idx is not None and up_idx not in blocked else 0.0
    can_down = 1.0 if down_idx is not None and down_idx not in blocked else 0.0
    can_left = 1.0 if left_idx is not None and left_idx not in blocked else 0.0
    can_right = 1.0 if right_idx is not None and right_idx not in blocked else 0.0
    return can_up, can_down, can_left, can_right


def featurize_state(state: GameState) -> np.ndarray:
    nearest_enemies = _nearest_enemies(state, k=2)
    nearest_projectiles = _nearest_projectiles(state, k=2)

    hp_ratio = min(max(state.hp / 12.0, 0.0), 1.0)

    def enemy_feats(enemy):
        if enemy is None:
            return 0.0, 0.0, 0.0, 0.0
        ex = (enemy.x - state.player_x) / 600.0
        ey = (enemy.y - state.player_y) / 600.0
        ed = min(sqrt((enemy.x - state.player_x) ** 2 + (enemy.y - state.player_y) ** 2) / 800.0, 1.0)
        es = min((abs(enemy.vx) + abs(enemy.vy)) / 10.0, 1.0)
        return ex, ey, ed, es

    def projectile_feats(proj):
        if proj is None:
            return 0.0, 0.0, 0.0, 0.0
        px = (proj.x - state.player_x) / 600.0
        py = (proj.y - state.player_y) / 600.0
        pvx = max(min(proj.vx / 12.0, 1.0), -1.0)
        pvy = max(min(proj.vy / 12.0, 1.0), -1.0)
        return px, py, pvx, pvy

    enemy1 = nearest_enemies[0] if len(nearest_enemies) >= 1 else None
    enemy2 = nearest_enemies[1] if len(nearest_enemies) >= 2 else None
    ex, ey, ed, es = enemy_feats(enemy1)
    ex2, ey2, ed2, es2 = enemy_feats(enemy2)

    proj1 = nearest_projectiles[0] if len(nearest_projectiles) >= 1 else None
    proj2 = nearest_projectiles[1] if len(nearest_projectiles) >= 2 else None
    px, py, pvx, pvy = projectile_feats(proj1)
    px2, py2, pvx2, pvy2 = projectile_feats(proj2)

    proj_threat1 = _incoming_projectile_threat(state, proj1)
    proj_threat2 = _incoming_projectile_threat(state, proj2)

    enemy_very_close = 1.0 if enemy1 is not None and ed < (42.0 / 800.0) else 0.0
    enemy_close = 1.0 if enemy1 is not None and ed < (72.0 / 800.0) else 0.0
    enemy_mid = 1.0 if enemy1 is not None and ed < (120.0 / 800.0) else 0.0

    shot_aligned = 0.0
    shot_blocked = 0.0
    if enemy1 is not None:
        if state.player_grid is not None and enemy1.grid is not None and state.grid_w > 0:
            same_col = (state.player_grid % state.grid_w) == (enemy1.grid % state.grid_w)
            same_row = (state.player_grid // state.grid_w) == (enemy1.grid // state.grid_w)
            if same_col or same_row:
                shot_aligned = 1.0
                blocked_for_shot = state.blocked | state.hazards | state.poops
                shot_blocked = 1.0 if _line_blocked_between_grids(
                    state.player_grid,
                    enemy1.grid,
                    state.grid_w,
                    blocked_for_shot,
                ) else 0.0

    can_up, can_down, can_left, can_right = _mobility_4(state)

    enemy_count = min(len(state.enemies), 6) / 6.0
    proj_count = min(len(state.projectiles), 8) / 8.0
    corner_flag = _in_corner_zone(state.player_grid, state.grid_w, state.grid_size)

    vec = np.array(
        [
            hp_ratio,
            ex,
            ey,
            ed,
            es,
            px,
            py,
            pvx,
            pvy,
            ex2,
            ey2,
            ed2,
            es2,
            px2,
            py2,
            pvx2,
            pvy2,
            proj_threat1,
            proj_threat2,
            enemy_very_close,
            enemy_close,
            enemy_mid,
            shot_aligned,
            shot_blocked,
            can_up,
            can_down,
            can_left,
            can_right,
            enemy_count,
            proj_count,
            corner_flag,
            1.0,
        ],
        dtype=np.float32,
    )
    return vec


def aim_direction(state: GameState) -> str:
    nearest_enemies = _nearest_enemies(state, k=1)
    enemy = nearest_enemies[0] if nearest_enemies else None
    if enemy is None:
        return "NONE"
    dx = enemy.x - state.player_x
    dy = enemy.y - state.player_y
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"
