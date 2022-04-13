includedirs { "../../contrib/Orochi/" }
files {"../../contrib/Orochi/Orochi/**.h", "../../contrib/Orochi/Orochi/**.cpp"}
files {"../../contrib/Orochi/contrib/**.h", "../../contrib/Orochi/contrib/**.cpp"}
if os.istarget("windows") then
    links{ "version" }
end

files { "../../hiprt/*.h"}
links {"hiprt64"}
libdirs{"../../hiprt/win/"}

print("depencency")