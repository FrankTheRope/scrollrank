# Rendering a tifxyz mesh without building VC3D

`vc_render_tifxyz` builds on its own, without Ceres and without the GUI. Verified
on Ubuntu 24 (WSL) against the official surface volume of PHerc1447 segment
235910: the central slice correlates at 0.998.

**The depth order comes out reversed.** With the command below, local slice *i*
matches official slice *30 − i* (r = 0.99 at every depth; same-index r ≈ 0.0
except at the centre). So villa's `--direction forward` on a local render is the
official volume's *reverse*, and vice versa. Every sweep in this repository ran
both directions and depth windows symmetric about the centre, so no result
depends on it, but a single-direction run on a local render reads the stack
upside down. Check the order against an official surface volume before trusting
one direction.

## Build (~20 min)

    sudo apt install -y build-essential cmake ninja-build git pkg-config \
      libopencv-dev libboost-all-dev libblosc-dev nlohmann-json3-dev libspdlog-dev \
      libtbb-dev zlib1g-dev libtiff-dev libgsl-dev libeigen3-dev \
      qt6-base-dev qt6-base-dev-tools libqt6opengl6-dev libgl1-mesa-dev \
      libcurl4-openssl-dev liblz4-dev libzstd-dev libcgal-dev libgmp-dev libmpfr-dev
    git clone --depth 1 https://github.com/ScrollPrize/villa.git && cd villa/volume-cartographer
    sed -i 's/^\(\s*\)find_package(Ceres/\1# find_package(Ceres/' CMakeLists.txt
    sed -i '/# find_package(Ceres REQUIRED)/a add_library(Ceres::ceres INTERFACE IMPORTED)' CMakeLists.txt
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_FLATBOI=OFF
    cmake --build build -t vc_render_tifxyz -j$(nproc)

Why: Ceres is a hard requirement at configure time but not for the renderer
target; core links `vc_lasagna` to `Ceres::ceres` unconditionally, hence the
dummy imported target; `VC_BUILD_FLATBOI=OFF` skips PaStiX (Fortran, BLAS,
LAPACKE, Scotch), which only the flattening tool needs. Thanks to Paul (VC team)
for the pointer.

## Render (streaming from S3, no volume download)

    aws s3 sync --no-sign-request s3://.../segments/<SEG>/mesh/tifxyz/ work/<SEG>.tifxyz/
    vc_render_tifxyz --volume work/cache/<scroll>.zarr \
      --remote-url s3://vesuvius-challenge-open-data/<SCROLL>/volumes/<VOL>.zarr/ \
      --segmentation work/<SEG>.tifxyz --scale 1 --group-idx 0 \
      --num-slices 31 --cache-gb 8 --voxel-size 8.64 --voxel-unit micrometer \
      --zarr-output work/<SEG>_render.zarr

One slice of a 3240x2980 segment took 14 min, almost all of it fetching chunks;
the chunk cache persists in work/cache so later renders are much faster. Use
`--zarr-output` for inference input, `--tif-output` to inspect slices.
