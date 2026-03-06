local myMod = RegisterMod("Isaac Relative Sensing", 1)
local socket = require("socket")

local client = nil
local shoot_dir = Vector(0, 0) 

function myMod:OnUpdate()
    -- 1. INITIALIZE CONNECTION
    if not client then
        client = socket.tcp()
        client:settimeout(0) 
        local success, err = client:connect("127.0.0.1", 5005)
        if success then print("LUA: Connected to Python Brain!") end
    end

    -- 2. THE EYES (Data Collection)
    if Game():GetFrameCount() % 2 == 0 and client then
        local player = Isaac.GetPlayer(0)
        local room = Game():GetRoom() -- <--- FIXED: Define room here
        local hp = player:GetHearts() + player:GetSoulHearts()
        
        -- Start the string with Player and HP
        local out = string.format("P:%.0f,%.0f|H:%d", player.Position.X, player.Position.Y, hp)
    
        -- SENSE DOORS
        for i = 0, 7 do
            local door = room:GetDoor(i)
            if door then
                local status = door:IsOpen() and "OPEN" or "CLOSED"
                out = out .. string.format("|D:%d:%s:%.0f,%.0f", i, status, door.Position.X, door.Position.Y)
            end
        end

        -- SENSE ENEMIES
        for _, ent in pairs(Isaac.GetRoomEntities()) do
            if ent:IsVulnerableEnemy() then
                out = out .. string.format("|E:%d.%d:%.0f,%.0f", ent.Type, ent.Variant, ent.Position.X, ent.Position.Y)
            end
        end

        -- Send to Python
        pcall(function() client:send(out .. "\n") end)
    end

    -- 3. THE HANDS (Action Execution)
    local data, err = client:receive("*l")
    if data then
        local player = Isaac.GetPlayer(0)
        if data:sub(1,5) == "MOVE:" then
            shoot_dir = Vector(0, 0)
            local move = data:sub(6)
            local speed = 5
            if move == "UP" then player.Velocity = Vector(0, -speed)
            elseif move == "DOWN" then player.Velocity = Vector(0, speed)
            elseif move == "LEFT" then player.Velocity = Vector(-speed, 0)
            elseif move == "RIGHT" then player.Velocity = Vector(speed, 0)
            elseif move == "STAY" then player.Velocity = player.Velocity * 0.8 -- Decelerate smoothly
            end
        
        elseif data:sub(1,6) == "SHOOT:" then
            local s = data:sub(7)
            if s == "UP" then shoot_dir = Vector(0, -1)
            elseif s == "DOWN" then shoot_dir = Vector(0, 1)
            elseif s == "LEFT" then shoot_dir = Vector(-1, 0)
            elseif s == "RIGHT" then shoot_dir = Vector(1, 0)
            else shoot_dir = Vector(0, 0) end
        end
    end
end

-- Force Isaac to shoot based on shoot_dir
myMod:AddCallback(ModCallbacks.MC_POST_PLAYER_RENDER, function(_, player)
    if shoot_dir:Length() > 0 then
        player:FireTear(player.Position, shoot_dir * 10, false, true, false)
    end
end)

myMod:AddCallback(ModCallbacks.MC_POST_UPDATE, myMod.OnUpdate)