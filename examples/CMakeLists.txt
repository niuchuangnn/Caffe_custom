file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)
    
  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
    
  add_executable(${name} ${source_file} ../include/caffe/layers/weightedSoftmax_loss_layer.hpp ../include/caffe/layers/pooling_switches_layer.hpp ../include/caffe/layers/slice_half_layer.hpp ../include/caffe/layers/inv_pooling_layer.hpp ../include/caffe/layers/superpixel_pooling_layer.hpp ../include/caffe/layers/unpooling_layer.hpp ../include/caffe/layers/superpixel_pooling1_layer.hpp ../include/caffe/layers/relu_mask_layer.hpp ../include/caffe/layers/select_seg_binary_layer.hpp ../include/caffe/layers/image_dim_prefetching_layer.hpp ../include/caffe/layers/bn_layer.hpp ../include/caffe/layers/conv_td_layer.hpp ../include/caffe/layers/inner_product_td_layer.hpp ../include/caffe/layers/pooling_td_layer.hpp ../include/caffe/layers/eltwise_td_layer.hpp)
  target_link_libraries(${name} ${Caffe_LINK})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION bin)

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standardized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()
