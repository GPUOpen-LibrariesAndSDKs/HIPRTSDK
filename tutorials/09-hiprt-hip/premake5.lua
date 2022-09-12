project "09-hiprt-hip"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/HipTestBase.h"} 
    files { "./**.h", "./**.cpp"} 

    includedirs{ "../../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"