local myMod = RegisterMod("Python AI Bridge", 1)
local socket = require("socket")

local client = nil

-- This function runs every single frame (60 times per second)
function myMod:OnUpdate()
    -- 1. INITIALIZE CONNECTION
    if not client then
        client = socket.tcp()
        client:settimeout(0) -- Critical: prevents game from freezing
        local success, err = client:connect("127.0.0.1", 5005)
        if success then 
            print("Connected to Python Brain!") 
        end
    end

    -- 2. THE EYES: Increased frequency to % 2 for smooth coordinate tracking
    if Game():GetFrameCount() % 2 == 0 and client then
        local player = Isaac.GetPlayer(0)
        -- Start with Player position
        local out = string.format("P:%.0f,%.0f", player.Position.X, player.Position.Y)
    
        -- Loop through everything in the room to find enemies
        for _, ent in pairs(Isaac.GetRoomEntities()) do
            if ent:IsVulnerableEnemy() then
                out = out .. string.format("|E:%.0f,%.0f", ent.Position.X, ent.Position.Y)
            end
        end

        -- Wrap in pcall so the game doesn't crash if Python is closed
        pcall(function() client:send(out .. "\n") end)
    end

    -- 3. THE HANDS: Check if Python sent a command
    local data, err = client:receive("*l")
    if data then
        local player = Isaac.GetPlayer(0)
        print("LUA RECEIVED: " .. tostring(data))
        -- The "spawn" command gives Isaac an item    
        if data == "spawn" then
            local pos = Isaac.GetFreeNearPosition(player.Position, 40)
            Isaac.Spawn(EntityType.ENTITY_PICKUP, PickupVariant.PICKUP_COLLECTIBLE, 0, pos, Vector(0,0), nil)
        
        -- The "hurt" command damages Isaac    
        elseif data == "hurt" then
            player:TakeDamage(1, 0, EntityRef(player), 0)
        -- The "speed" command increases Isaacs movement speed
        elseif data == "speed" then
            player.MoveSpeed = player.MoveSpeed + 0.2
        -- The "tp" command teleports Isaac to the center of the room    
        elseif data == "tp" then
            player.Position = Vector(320, 280)
            print("Action: Teleported to center")
        end
    end
end

myMod:AddCallback(ModCallbacks.MC_POST_UPDATE, myMod.OnUpdate)

print("Python AI Bridge: Mod Loaded!")
