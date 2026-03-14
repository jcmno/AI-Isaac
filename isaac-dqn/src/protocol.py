from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Enemy:
    enemy_id: str
    x: float
    y: float
    vx: float
    vy: float
    grid: Optional[int]


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    grid: Optional[int]


@dataclass
class Door:
    slot: int
    status: str
    x: float
    y: float
    grid: Optional[int]
    is_curse: bool = False


@dataclass
class Item:
    item_type: int
    variant: int
    x: float
    y: float
    grid: Optional[int]
    price: int


@dataclass
class Button:
    x: float
    y: float
    grid: Optional[int]


@dataclass
class GameState:
    player_x: float
    player_y: float
    player_grid: Optional[int]
    hp: int
    red_hearts: int
    max_red_hearts: int
    room_key: str
    coins: int
    enemies: list[Enemy]
    projectiles: list[Projectile]
    doors: list[Door]
    items: list[Item]
    grid_w: int
    grid_size: int
    blocked: set[int]
    hazards: set[int]
    poops: set[int]
    buttons: list[Button]


def _parse_int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_packet(line: str) -> Optional[GameState]:
    parts = line.strip().split("|")
    if not parts:
        return None

    player_x = 0.0
    player_y = 0.0
    player_grid: Optional[int] = None
    hp = 0
    red_hearts = 0
    max_red_hearts = 0
    room_key = "UNKNOWN"
    coins = 0

    enemies: list[Enemy] = []
    projectiles: list[Projectile] = []
    doors: list[Door] = []
    items: list[Item] = []
    buttons: list[Button] = []
    poops: set[int] = set()
    grid_w = 0
    grid_size = 0
    blocked: set[int] = set()
    hazards: set[int] = set()

    for p in parts:
        if p.startswith("P:"):
            p_info = p[2:].split(",")
            if len(p_info) >= 2:
                player_x = _parse_float(p_info[0])
                player_y = _parse_float(p_info[1])
            if len(p_info) >= 3:
                player_grid = _parse_int(p_info[2], default=-1)
                if player_grid < 0:
                    player_grid = None

        elif p.startswith("H:"):
            hp = _parse_int(p[2:])

        elif p.startswith("V:"):
            v_info = p.split(":")
            if len(v_info) >= 3:
                red_hearts = _parse_int(v_info[1], default=0)
                max_red_hearts = _parse_int(v_info[2], default=0)

        elif p.startswith("R:"):
            r_info = p.split(":")
            if len(r_info) >= 3:
                room_key = f"{r_info[1]}:{r_info[2]}"

        elif p.startswith("C:"):
            coins = _parse_int(p[2:])

        elif p.startswith("E:"):
            e_parts = p.split(":")
            if len(e_parts) < 3:
                continue
            pos = e_parts[2].split(",")
            vx = 0.0
            vy = 0.0
            if len(e_parts) >= 4:
                vel = e_parts[3].split(",")
                if len(vel) >= 2:
                    vx = _parse_float(vel[0])
                    vy = _parse_float(vel[1])
            e_grid = None
            if len(e_parts) >= 5:
                g = _parse_int(e_parts[4], default=-1)
                e_grid = g if g >= 0 else None
            enemies.append(
                Enemy(
                    enemy_id=e_parts[1],
                    x=_parse_float(pos[0]) if len(pos) >= 1 else 0.0,
                    y=_parse_float(pos[1]) if len(pos) >= 2 else 0.0,
                    vx=vx,
                    vy=vy,
                    grid=e_grid,
                )
            )

        elif p.startswith("T:"):
            t_parts = p.split(":")
            if len(t_parts) < 2:
                continue
            pos = t_parts[1].split(",")
            vx = 0.0
            vy = 0.0
            if len(t_parts) >= 3:
                vel = t_parts[2].split(",")
                if len(vel) >= 2:
                    vx = _parse_float(vel[0])
                    vy = _parse_float(vel[1])
            t_grid = None
            if len(t_parts) >= 4:
                g = _parse_int(t_parts[3], default=-1)
                t_grid = g if g >= 0 else None
            projectiles.append(
                Projectile(
                    x=_parse_float(pos[0]) if len(pos) >= 1 else 0.0,
                    y=_parse_float(pos[1]) if len(pos) >= 2 else 0.0,
                    vx=vx,
                    vy=vy,
                    grid=t_grid,
                )
            )

        elif p.startswith("D:"):
            d_parts = p.split(":")
            if len(d_parts) < 4:
                continue
            coords = d_parts[3].split(",")
            d_grid = None
            if len(d_parts) >= 5:
                g = _parse_int(d_parts[4], default=-1)
                d_grid = g if g >= 0 else None
            d_is_curse = False
            if len(d_parts) >= 6:
                d_is_curse = _parse_int(d_parts[5], default=0) == 1
            doors.append(
                Door(
                    slot=_parse_int(d_parts[1], default=-1),
                    status=d_parts[2],
                    x=_parse_float(coords[0]) if len(coords) >= 1 else 0.0,
                    y=_parse_float(coords[1]) if len(coords) >= 2 else 0.0,
                    grid=d_grid,
                    is_curse=d_is_curse,
                )
            )

        elif p.startswith("I:"):
            i_parts = p.split(":")
            if len(i_parts) < 3:
                continue
            i_type = 0
            i_variant = 0
            if "." in i_parts[1]:
                left, right = i_parts[1].split(".", 1)
                i_type = _parse_int(left)
                i_variant = _parse_int(right)
            pos = i_parts[2].split(",")
            i_grid = None
            if len(i_parts) >= 4:
                g = _parse_int(i_parts[3], default=-1)
                i_grid = g if g >= 0 else None
            price = 0
            if len(i_parts) >= 5:
                price = _parse_int(i_parts[4], default=0)
            items.append(
                Item(
                    item_type=i_type,
                    variant=i_variant,
                    x=_parse_float(pos[0]) if len(pos) >= 1 else 0.0,
                    y=_parse_float(pos[1]) if len(pos) >= 2 else 0.0,
                    grid=i_grid,
                    price=price,
                )
            )

        elif p.startswith("G:"):
            g_info = p.split(":", 3)
            if len(g_info) >= 3:
                grid_w = _parse_int(g_info[1])
                grid_size = _parse_int(g_info[2])
            if len(g_info) == 4 and g_info[3]:
                blocked = {int(x) for x in g_info[3].split(",") if x.strip()}

        elif p.startswith("K:"):
            k_info = p.split(":", 1)
            if len(k_info) == 2 and k_info[1]:
                poops = {int(x) for x in k_info[1].split(",") if x.strip()}

        elif p.startswith("B:"):
            b_parts = p.split(":")
            if len(b_parts) >= 3:
                bcoords = b_parts[1].split(",")
                b_grid = _parse_int(b_parts[2], default=-1)
                buttons.append(
                    Button(
                        x=_parse_float(bcoords[0]) if len(bcoords) >= 1 else 0.0,
                        y=_parse_float(bcoords[1]) if len(bcoords) >= 2 else 0.0,
                        grid=b_grid if b_grid >= 0 else None,
                    )
                )

        elif p.startswith("Z:"):
            z_info = p.split(":", 1)
            if len(z_info) == 2 and z_info[1]:
                hazards = {int(x) for x in z_info[1].split(",") if x.strip()}

    return GameState(
        player_x=player_x,
        player_y=player_y,
        player_grid=player_grid,
        hp=hp,
        red_hearts=red_hearts,
        max_red_hearts=max_red_hearts,
        room_key=room_key,
        coins=coins,
        enemies=enemies,
        projectiles=projectiles,
        doors=doors,
        items=items,
        grid_w=grid_w,
        grid_size=grid_size,
        blocked=blocked,
        hazards=hazards,
        poops=poops,
        buttons=buttons,
    )
