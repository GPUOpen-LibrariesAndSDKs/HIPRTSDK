
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
      defines {"TH_DEBUG_GPU"}
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
    buildoptions "-std=c++11"


    include "00_context_creation"
    include "01_geom_intersection"
	include "02_scene_intersection"
	include "03_custom_intersection"
	include "04_shared_stack"
	include "05_custom_bvh_import"
	include "06_obj_AO"
    include "07_motion_blur"
	include "08_multi_custom_intersection"
	include "09-hiprt-hip"
	
    if os.ishost("windows") then
		local hiproot = os.getenv("HIP_PATH"):gsub([[\]],[[/]])
        copydir( "../hiprt/win/", "./build/" )
		copydir( "../contrib/Orochi/contrib/bin/win64", "./build/" )
		if hiproot ~= nil then
			copydir( hiproot .."/bin/", "./build/", "amdhip64.dll" )
		end
    end
