
local generating = _ACTION and _ACTION ~= "clean" and not _OPTIONS["help"] and not _OPTIONS["version"]

if generating then
    local root = path.getabsolute("..", _SCRIPT_DIR)
    local function setup(command)
        local previous = os.getcwd()
        os.chdir(root)
        local ok = os.execute(command)
        os.chdir(previous)
        if not ok then
            error("Dependency setup failed. Resolve the error above and run Premake again.", 0)
        end
    end

    local orochi = path.join(root, "contrib/Orochi/Orochi/Orochi.h")
    if not os.isfile(orochi) then
        print("Initializing tutorial submodules...")
        setup("git submodule update --init --recursive")
        if not os.isfile(orochi) then
            error("Tutorial submodules are incomplete. Restore contrib/Orochi and run Premake again.", 0)
        end
    end

    local header = path.join(root, "hiprt/hiprt.h")
    -- Respect any HIPRT installation supplied by the user.
    if not os.isfile(header) then
        print("hiprt/hiprt.h is missing; building HIP RT from source...")
        if os.ishost("windows") then
            setup('call "' .. path.translate(path.join(root, "build_hiprt.bat"), "\\") .. '"')
        elseif os.ishost("linux") then
            local script = path.join(root, "build_hiprt.sh"):gsub("'", "'\\''")
            setup("bash '" .. script .. "'")
        else
            error("Automatic HIPRT builds support Windows and Linux. Install HIPRT in hiprt/ first.", 0)
        end
        if not os.isfile(header) then
            error("HIPRT setup did not install hiprt/hiprt.h. Run the build script manually to diagnose the failure.", 0)
        end
    end

    local file = assert(io.open(header, "r"))
    local version = file:read("*a"):match('#define%s+HIPRT_VERSION_STR%s+"(%d+)"')
    file:close()
    if not version then
        error("Cannot read HIPRT_VERSION_STR from hiprt/hiprt.h.", 0)
    end
    HIPRT_LIBRARY = "hiprt" .. version .. "64"
end

function copydir(src_dir, dst_dir, filter, single_dst_dir)
    filter = filter or "**"
    src_dir = src_dir .. "/"
    print('copy "' .. src_dir .. filter .. '" to "' .. dst_dir .. '".')
    dst_dir = dst_dir .. "/"
    local dir = path.rebase(".", path.getabsolute("."), src_dir) -- root dir, relative from src_dir

    os.chdir(src_dir) -- change current directory to src_dir
    local matches = os.matchfiles(filter)
    os.chdir(dir) -- change current directory back to root

    local counter = 0
    for k, v in ipairs(matches) do
        local target = iif(single_dst_dir, path.getname(v), v)
        --make sure, that directory exists or os.copyfile() fails
        os.mkdir(path.getdirectory(dst_dir .. target))
        if os.copyfile(src_dir .. v, dst_dir .. target) then
            counter = counter + 1
        end
    end

    if counter == #matches then
        print(counter .. " files copied.")
        return true
    else
        print("Error: " .. counter .. "/" .. #matches .. " files copied.")
        return nil
    end
end

workspace "hiprtSdkTutorial"
    configurations {"Debug", "Release", "RelWithDebInfo", "DebugGpu" }
    language "C++"
    platforms "x64"
    architecture "x86_64"

    if os.ishost("windows") then
        defines {"__WINDOWS__"}
    end
    characterset("MBCS")

    filter {"platforms:x64", "configurations:Debug or configurations:DebugGpu"}
      targetsuffix "64D"
      defines {"DEBUG"}
      symbols "On"
    filter {"platforms:x64", "configurations:DebugGpu"}
      defines {"DEBUG_GPU"}
    filter {"platforms:x64", "configurations:Release or configurations:RelWithDebInfo"}
      targetsuffix "64"
      defines {"NDEBUG"}
      optimize "On"
    filter {"platforms:x64", "configurations:RelWithDebInfo"}
      symbols "On"
    filter {}

    if os.ishost("windows") then
        buildoptions {"/wd4244", "/wd4305", "/wd4018", "/wd4996"}
    end

    include "00_context_creation"
    include "01_geom_intersection"
    include "02_scene_intersection"
    include "03_custom_intersection"
    include "04_compaction"
    include "05_global_stack"
    include "06_dynamic_stack"
    include "07_custom_bvh_import"
    include "08_ambient_occlusion"
    include "09_motion_blur_srt"
    include "10_motion_blur_matrix"
    include "11_multi_custom_intersection"
    include "12_cutout"
    include "13_concurrent_scene_build"
    include "14_batch_build"
    include "15_multi_level_instancing"
    include "16_fluid_simulation"
    include "17_hiprt_hip"
    include "18_shadow_ray"
    include "19_primary_ray"

    if os.ishost("windows") and generating then
		  copydir( "../hiprt/win/", "./dist/bin/Release/" )
      copydir( "../hiprt/win/", "./dist/bin/Debug/" )
    end
