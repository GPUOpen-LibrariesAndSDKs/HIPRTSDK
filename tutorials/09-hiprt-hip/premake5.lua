project "09-hiprt-hip"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/HipTestBase.h"} 
    files { "./**.h", "./**.cpp"} 
	
    includedirs{ "../../" } 
	
	local hiproot = os.getenv("HIP_PATH"):gsub([[\]],[[/]])
	libdirs { hiproot .. "/lib/" }
    links { "amdhip64" }
	
    targetdir "../dist/bin/%{cfg.buildcfg}"