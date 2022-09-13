project "09-hiprt-hip"
    kind "ConsoleApp"
    location "../build"
  
	local hiproot = os.getenv("HIP_PATH"):gsub([[\]],[[/]])
	if hiproot ~= nil then
		includedirs { hiproot .. "/include/" }
	end

	includedirs { "../../" }
	
    files { "../common/HipTestBase.h"} 
    files { "./**.h", "./**.cpp"} 
	files { "../../hiprt/*.h"}
	
if os.istarget("windows") then
	if hiproot ~= nil then
		libdirs { hiproot .. "/lib/"}
		links { "amdhip64" }
	end
	libdirs {"../../hiprt/win/" }
end
if os.ishost("linux") then
	if hiproot ~= nil then
		libdirs { hiproot .. "/lib/"}
		links { "libamdhip64" }
	end
	libdirs { "../../hiprt/linux64/" }
end
	
links {"hiprt64"}
targetdir "../dist/bin/%{cfg.buildcfg}"