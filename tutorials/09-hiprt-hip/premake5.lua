project "09-hiprt-hip"
    kind "ConsoleApp"
    location "../build"
  
	local hiproot = os.getenv("HIP_PATH"):gsub([[\]],[[/]])
	includedirs { hiproot .. "/include/", "../../" }

    files { "../common/HipTestBase.h"} 
    files { "./**.h", "./**.cpp"} 
	files { "../../hiprt/*.h"}
	
if os.istarget("windows") then
	libdirs { hiproot .. "/lib/", "../../hiprt/win/" }
	links { "amdhip64" }
end
if os.ishost("linux") then
	libdirs { hiproot .. "/lib/", "../../hiprt/linux64/" }
    links { "libamdhip64" }
end
	
links {"hiprt64"}
targetdir "../dist/bin/%{cfg.buildcfg}"