project "06_obj_AO"
    kind "ConsoleApp"
    location "../build"

    dofile "../common/dependency.lua"

    files { "../common/**.h", "../common/**.cpp"} 
    files { "./**.h", "./**.cpp"} 

    includedirs{ "../../", "../" } 

    targetdir "../dist/bin/%{cfg.buildcfg}"