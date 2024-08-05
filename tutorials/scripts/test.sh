PWD_PATH=`pwd`
export LD_LIBRARY_PATH="$PWD_PATH/../../hiprt/linux64:${LD_LIBRARY_PATH}"

../dist/bin/Release/00_context_creation64
../dist/bin/Release/01_geom_intersection64
../dist/bin/Release/02_scene_intersection64
../dist/bin/Release/03_custom_intersection64
../dist/bin/Release/04_compaction64
../dist/bin/Release/05_global_stack64
../dist/bin/Release/06_dynamic_stack64
../dist/bin/Release/07_custom_bvh_import64
../dist/bin/Release/08_ambient_occlusion64
../dist/bin/Release/09_motion_blur_srt64
../dist/bin/Release/10_motion_blur_matrix64
../dist/bin/Release/11_multi_custom_intersection64
../dist/bin/Release/12_cutout64
../dist/bin/Release/13_concurrent_scene_build64
../dist/bin/Release/14_batch_build64
../dist/bin/Release/15_multi_level_instancing64
../dist/bin/Release/16_fluid_simulation64
../dist/bin/Release/18_shadow_ray64
../dist/bin/Release/19_primary_ray64
