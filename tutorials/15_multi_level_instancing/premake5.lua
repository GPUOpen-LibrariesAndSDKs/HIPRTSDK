project "15_multi_level_instancing"
    cppdialect "C++17"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/**.h", "../common/**.cpp"} 
    files { "./**.h", "./**.cpp"} 

    includedirs{ "../../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"