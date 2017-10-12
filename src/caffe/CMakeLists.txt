# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
caffe_default_properties(proto)
target_link_libraries(proto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(proto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Caffe_LINKER_LIBS 0 PUBLIC proto) # note, crucial to prepend!

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs} layers/weightedSoftmax_loss_layer.cpp layers/pooling_switches_layer.cpp layers/slice_half_layer.cpp layers/inv_pooling_layer.cpp layers/superpixel_pooling_layer.cpp layers/unpooling_layer.cpp layers/superpixel_pooling1_layer.cpp layers/relu_mask_layer.cpp layers/relu_mask_layer.cu layers/select_seg_binary_layer.cpp layers/image_dim_prefetching_layer.cpp layers/image_dim_prefetching_layer.cu layers/bn_layer.cpp layers/bn_layer.cu layers/conv_td_layer.cpp layers/inner_product_td_layer.cpp layers/pooling_td_layer.cpp layers/eltwise_td_layer.cpp layers/inner_product_td_layer.cu layers/pooling_td_layer.cu layers/eltwise_td_layer.cu layers/conv_td_layer.cu)
caffe_default_properties(caffe)
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
target_include_directories(caffe ${Caffe_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)
target_compile_definitions(caffe ${Caffe_DEFINITIONS})
if(Caffe_COMPILE_OPTIONS)
  target_compile_options(caffe ${Caffe_COMPILE_OPTIONS})
endif()
set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)

