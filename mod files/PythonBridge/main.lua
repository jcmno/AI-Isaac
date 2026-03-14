local myMod = RegisterMod("Isaac Relative Sensing", 1)
local socket = require("socket")

local client = nil
local shoot_dir = Vector(0, 0) 
local shoot_cooldown = 0

local function apply_command(player, cmd)
    if cmd:sub(1,5) == "MOVE:" then
        shoot_dir = Vector(0, 0)
        local move = cmd:sub(6)
        local speed = 5
        if move == "UP" then player.Velocity = Vector(0, -speed)
        elseif move == "DOWN" then player.Velocity = Vector(0, speed)
        elseif move == "LEFT" then player.Velocity = Vector(-speed, 0)
        elseif move == "RIGHT" then player.Velocity = Vector(speed, 0)
        elseif move == "STAY" then player.Velocity = player.Velocity * 0.8 -- Decelerate smoothly
        end

    elseif cmd:sub(1,6) == "SHOOT:" then
        local s = cmd:sub(7)
        if s == "UP" then shoot_dir = Vector(0, -1)
        elseif s == "DOWN" then shoot_dir = Vector(0, 1)
        elseif s == "LEFT" then shoot_dir = Vector(-1, 0)
        elseif s == "RIGHT" then shoot_dir = Vector(1, 0)
        else shoot_dir = Vector(0, 0) end

    elseif cmd == "RESTART" then
        -- Restart the game after death
        Isaac.ExecuteCommand("restart")
    end
end

function myMod:OnUpdate()
    if shoot_cooldown > 0 then
        shoot_cooldown = shoot_cooldown - 1
    end

    -- 1. INITIALIZE CONNECTION
    if not client then
        client = socket.tcp()
        client:settimeout(0) 
        local success, err = client:connect("127.0.0.1", 5005)
        if success then print("LUA: Connected to Python Brain!") end
    end

    -- 2. THE EYES (Data Collection)
    if client then
        local player = Isaac.GetPlayer(0)
        local game = Game()
        local room = game:GetRoom()
        local level = game:GetLevel()
        local room_desc = level:GetCurrentRoomDesc()
        local red_hearts = player:GetHearts()
        local max_red_hearts = player:GetMaxHearts()
        local hp = player:GetHearts() + player:GetSoulHearts()
        local coins = player:GetNumCoins()
        local player_grid = room:GetGridIndex(player.Position)
        
        -- Packet format:
        -- P:x,y,g        -> player position + player grid index
        -- H:hp           -> total current health
        -- V:red:max_red  -> red-heart status for heart-pickup filtering
        -- C:coins        -> current coin count for store affordability checks
        -- R:stage:room   -> coarse room identity for transition-aware logic in Python
        local out = string.format(
            "P:%.0f,%.0f,%d|H:%d|V:%d:%d|C:%d|R:%d:%d",
            player.Position.X,
            player.Position.Y,
            player_grid,
            hp,
            red_hearts,
            max_red_hearts,
            coins,
            level:GetStage(),
            room_desc.SafeGridIndex
        )
    
        -- D:slot:status:x,y:g  (status is OPEN/CLOSED)
        -- Slot id is important so Python can avoid immediately taking the opposite door.
        for i = 0, 7 do
            local door = room:GetDoor(i)
            if door then
                local status = door:IsOpen() and "OPEN" or "CLOSED"
                local door_grid = room:GetGridIndex(door.Position)
                local is_curse = (door.TargetRoomType == RoomType.ROOM_CURSE) and 1 or 0
                out = out .. string.format("|D:%d:%s:%.0f,%.0f:%d:%d", i, status, door.Position.X, door.Position.Y, door_grid, is_curse)
            end
        end

        -- G:width:size:blocked_indices_csv
        -- Z:hazard_indices_csv
        -- K:poop_indices_csv  (destructible by shooting)
        -- Send walkability plus damaging tiles (e.g., fire) so Python can avoid self-damage.
        local grid_w = room:GetGridWidth()
        local grid_size = room:GetGridSize()
        local blocked = {}
        local hazards = {}
        local poops = {}
        for idx = 0, grid_size - 1 do
            local coll = room:GetGridCollision(idx)
            if coll ~= 0 then
                blocked[#blocked + 1] = tostring(idx)
            end

            local ge = room:GetGridEntity(idx)
            if ge and ge.Desc then
                local gt = ge.Desc.Type
                if gt == GridEntityType.GRID_FIREPLACE or gt == GridEntityType.GRID_SPIKES then
                    hazards[#hazards + 1] = tostring(idx)
                elseif gt == GridEntityType.GRID_POOP then
                    poops[#poops + 1] = tostring(idx)
                elseif gt == GridEntityType.GRID_PRESSURE_PLATE then
                    -- B:x,y:g  -> pressure-plate button position
                    local bpos = room:GetGridPosition(idx)
                    out = out .. string.format("|B:%.0f,%.0f:%d", bpos.X, bpos.Y, idx)
                end
            end
        end
        out = out .. string.format("|G:%d:%d:%s", grid_w, grid_size, table.concat(blocked, ","))
        out = out .. string.format("|K:%s", table.concat(poops, ","))
        out = out .. string.format("|Z:%s", table.concat(hazards, ","))

        -- E:type.variant:x,y:vx,vy:g
        -- I:type.variant:x,y:g:price
        -- T:x,y:vx,vy:g   (enemy projectile telemetry)
        -- We include enemy velocity for threat assessment, pickups for looting, and projectiles for dodging.
        for _, ent in pairs(Isaac.GetRoomEntities()) do
            if ent:IsVulnerableEnemy()
                and not ent:HasEntityFlags(EntityFlag.FLAG_FRIENDLY)
                and ent.Type ~= EntityType.ENTITY_FAMILIAR
                and ent.Type ~= EntityType.ENTITY_PLAYER then
                local e_grid = room:GetGridIndex(ent.Position)
                out = out .. string.format(
                    "|E:%d.%d:%.0f,%.0f:%.2f,%.2f:%d",
                    ent.Type,
                    ent.Variant,
                    ent.Position.X,
                    ent.Position.Y,
                    ent.Velocity.X,
                    ent.Velocity.Y,
                    e_grid
                )
            elseif ent.Type == 5 then -- pickups (coins/hearts/keys/items/etc)
                local i_grid = room:GetGridIndex(ent.Position)
                local price = ent.Price or 0
                out = out .. string.format("|I:%d.%d:%.0f,%.0f:%d:%d", ent.Type, ent.Variant, ent.Position.X, ent.Position.Y, i_grid, price)
            elseif ent.Type == 9 then -- enemy projectiles
                local t_grid = room:GetGridIndex(ent.Position)
                out = out .. string.format("|T:%.0f,%.0f:%.2f,%.2f:%d", ent.Position.X, ent.Position.Y, ent.Velocity.X, ent.Velocity.Y, t_grid)
            end
        end

        -- pcall keeps the game loop alive if the socket hiccups.
        pcall(function() client:send(out .. "\n") end)
    end

    -- 3. THE HANDS (Action Execution)
    local data, err = client:receive("*l")
    if data then
        local player = Isaac.GetPlayer(0)
        -- Supports either a single command (legacy) or combined commands: MOVE:...;SHOOT:...
        for cmd in string.gmatch(data, "[^;]+") do
            apply_command(player, cmd)
        end
    end
end

-- Force Isaac to shoot based on shoot_dir
myMod:AddCallback(ModCallbacks.MC_POST_PLAYER_RENDER, function(_, player)
    if shoot_dir:Length() > 0 and shoot_cooldown <= 0 then
        player:FireTear(player.Position, shoot_dir * 10, false, true, false)
        shoot_cooldown = 8
    end
end)

myMod:AddCallback(ModCallbacks.MC_POST_UPDATE, myMod.OnUpdate)