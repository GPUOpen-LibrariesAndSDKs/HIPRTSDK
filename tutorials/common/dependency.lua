local hiproot = os.getenv("HIP_PATH"):gsub([[\]],[[/]])
includedirs { hiproot .. "/include/", "../../contrib/Orochi/" }

if os.istarget("windows") then
    links{ "version" }
    libdirs{"../../hiprt/win/"}
	libdirs { hiproot .. "/lib/" }
    links { "amdhip64" }
end
if os.ishost("linux") then
    links { "pthread", "dl" }
    libdirs{"../../hiprt/linux64/"}
end

files { "../../hiprt/*.h"}
links {"hiprt64"}

print("depencency")