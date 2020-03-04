#include "feature/sift.h"

#include <array>
#include <fstream>
#include <memory>

#include "FLANN/flann.hpp"
#include "SiftGPU/SiftGPU.h"
#include "VLFeat/covdet.h"
#include "VLFeat/sift.h"
#include "feature/utils.h"
#include "util/cuda.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/option_manager.h"
#include "util/threading.h"
#include "util/misc.h"
#include "feature/extraction.h"

#undef slots

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <mutex>

#include <numeric>

using namespace colmap;


namespace py = pybind11;

typedef py::array_t<uint8_t, py::array::c_style> ndarray_uint8;

std::shared_ptr<SiftExtractionOptions> make_deafault_options()
{
	OptionManager options;
	//options.ModifyForExtremeQuality();
    options.AddExtractionOptions();
    return options.sift_extraction;
}

Bitmap Resize(Bitmap bitmap, const std::shared_ptr<SiftExtractionOptions>& options)
{
	if (static_cast<int>(bitmap.Width()) > options->max_image_size || static_cast<int>(bitmap.Height()) > options->max_image_size)
	{
	    // Fit the down-sampled version exactly into the max dimensions.
	    const double scale = static_cast<double>(options->max_image_size) / std::max(bitmap.Width(), bitmap.Height());
	    const int new_width =static_cast<int>(bitmap.Width() * scale);
	    const int new_height = static_cast<int>(bitmap.Height() * scale);
	    bitmap.Rescale(new_width, new_height);
		return bitmap;
	}
	return bitmap;
}

class SiftExtractor
{
public:
	SiftExtractor(const std::shared_ptr<SiftExtractionOptions>& options)
	{
	    CHECK(options->Check());
		m_options = *options;
		const int num_threads = GetEffectiveNumThreads(options->num_threads);
		CHECK_GT(num_threads, 0);

		CHECK_EQ(options->domain_size_pooling, false);
		CHECK_EQ(options->estimate_affine_shape, false);
		CHECK_EQ(options->use_gpu, true);

	    std::vector<int> gpu_indices = CSVToVector<int>(options->gpu_index);

	    if (gpu_indices.size() == 1 && gpu_indices[0] == -1)
	    {
	        const int num_cuda_devices = GetNumCudaDevices();
	        CHECK_GT(num_cuda_devices, 0);
	        gpu_indices.resize(num_cuda_devices);
	        std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
	    }

	    sift_gpu.reset(new SiftGPU);
	    if (!CreateSiftGPUExtractor(*options.get(), sift_gpu.get())) {
	        std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
	        return;
	    }
		CHECK_EQ(m_options.max_image_size, sift_gpu->GetMaxDimension());
		CHECK(!m_options.estimate_affine_shape);
		CHECK(!m_options.domain_size_pooling);
	}

	std::pair<FeatureDescriptors, FeatureKeypoints> ExtractSiftFeaturesGPU(const Bitmap& bitmap)
	{
		CHECK(bitmap.IsGrey());

		// Note, that this produces slightly different results than using SiftGPU
		// directly for RGB->GRAY conversion, since it uses different weights.
		const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
		const int code =
				sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
				                  bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

		const int kSuccessCode = 1;
		if (code != kSuccessCode)
		{
			LOG(FATAL) << "Run SIFT did nit succeed";
		}

		const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());

		std::vector<SiftKeypoint> keypoints_data(num_features);

		// Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
				descriptors_float(num_features, 128);

		// Download the extracted keypoints and descriptors.
		sift_gpu->GetFeatureVector(keypoints_data.data(), descriptors_float.data());

	    FeatureKeypoints keypoints;
	    keypoints.resize(num_features);
		for (size_t i = 0; i < num_features; ++i) {
			keypoints[i] = FeatureKeypoint(keypoints_data[i].x, keypoints_data[i].y,
		                                   keypoints_data[i].s, keypoints_data[i].o);
		}

		// Save and normalize the descriptors.
		if (m_options.normalization == SiftExtractionOptions::Normalization::L2)
		{
			descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
		}
		else if (m_options.normalization == SiftExtractionOptions::Normalization::L1_ROOT)
		{
			descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
		}
		else
		{
			LOG(FATAL) << "Normalization type not supported";
		}

		FeatureDescriptors descriptors = FeatureDescriptorsToUnsignedByte(descriptors_float);

		return std::make_pair(descriptors, keypoints);
	}

	void SetKeyPoints(const FeatureKeypoints& keypoints, bool skip_orientation)
	{
		std::vector<SiftKeypoint> keypoints_data(keypoints.size());

		for (size_t i = 0; i < keypoints.size(); ++i) {
			keypoints_data[i].x = keypoints[i].x;
			keypoints_data[i].y = keypoints[i].y;
			keypoints_data[i].s = keypoints[i].ComputeScale();
			keypoints_data[i].o = keypoints[i].ComputeOrientation();
		}

		sift_gpu->SetKeypointList(keypoints.size(), keypoints_data.data(), skip_orientation ? 1 : 0);
	}

	SiftExtractionOptions m_options;
	std::shared_ptr<SiftGPU> sift_gpu;
};

Bitmap CreateFromNumpy(ndarray_uint8 ndarray)
{
	const py::buffer_info& ndarray_info = ndarray.request();

	CHECK_EQ(ndarray_info.ndim, 3);

	int width = ndarray_info.shape[1];
	int height = ndarray_info.shape[0];

	CHECK_EQ(ndarray_info.shape[2] == 3 || ndarray_info.shape[2] == 1, true);

	FIBITMAP* bmp = nullptr;
	if (ndarray_info.shape[2] == 1)
	{
		bmp = FreeImage_ConvertFromRawBits((BYTE*)ndarray_info.ptr, width, height, width * sizeof(uint8_t), 8, 0, 0, 0, true);
	}
	else if (ndarray_info.shape[2] == 3)
	{
		bmp = FreeImage_ConvertFromRawBits((BYTE*)ndarray_info.ptr, width, height, width * 3 * sizeof(uint8_t), 24, 0, 0, 0, true);
	}
	return Bitmap(bmp);
}


PYBIND11_MODULE(siftgpu, m) {
	m.doc() = "siftgpu";

    m.def("make_default_quality_options", &make_deafault_options, "Returns default quality settings for feature extraction");

    py::class_<SiftExtractionOptions, std::shared_ptr<SiftExtractionOptions> >(m, "SiftExtractionOptions")
		.def(py::init<>())
        .def_readwrite("num_threads", &SiftExtractionOptions::num_threads)
        .def_readwrite("use_gpu", &SiftExtractionOptions::use_gpu)
        .def_readwrite("gpu_index", &SiftExtractionOptions::gpu_index)
        .def_readwrite("max_image_size", &SiftExtractionOptions::max_image_size)
        .def_readwrite("max_num_features", &SiftExtractionOptions::max_num_features)
        .def_readwrite("first_octave", &SiftExtractionOptions::first_octave)
        .def_readwrite("num_octaves", &SiftExtractionOptions::num_octaves)
        .def_readwrite("octave_resolution", &SiftExtractionOptions::octave_resolution)
        .def_readwrite("peak_threshold", &SiftExtractionOptions::peak_threshold)
        .def_readwrite("edge_threshold", &SiftExtractionOptions::edge_threshold)
        .def_readwrite("estimate_affine_shape", &SiftExtractionOptions::estimate_affine_shape)
        .def_readwrite("max_num_orientations", &SiftExtractionOptions::max_num_orientations)
        .def_readwrite("upright", &SiftExtractionOptions::upright)
        .def_readwrite("estimate_affine_shape", &SiftExtractionOptions::estimate_affine_shape)
        .def_readwrite("darkness_adaptivity", &SiftExtractionOptions::darkness_adaptivity)
        .def_readwrite("domain_size_pooling", &SiftExtractionOptions::domain_size_pooling)
        .def_readwrite("dsp_min_scale", &SiftExtractionOptions::dsp_min_scale)
        .def_readwrite("dsp_max_scale", &SiftExtractionOptions::dsp_max_scale)
        .def_readwrite("dsp_num_scales", &SiftExtractionOptions::dsp_num_scales)
        .def_readwrite("normalization", &SiftExtractionOptions::normalization)
        ;

	py::enum_<SiftExtractionOptions::Normalization>(m, "SiftNormalization")
		.value("L1_ROOT", SiftExtractionOptions::Normalization::L1_ROOT)
		.value("L2", SiftExtractionOptions::Normalization::L2)
		.export_values();

	py::enum_<FREE_IMAGE_FORMAT>(m, "FREE_IMAGE_FORMAT")
		.value("FIF_UNKNOWN", FIF_UNKNOWN)
		.value("FIF_BMP", FIF_BMP)
		.value("FIF_JPEG", FIF_JPEG)
		.value("FIF_PGM", FIF_PGM)
		.value("FIF_PNG", FIF_PNG)
		.value("FIF_TARGA", FIF_TARGA)
		.value("FIF_TIFF", FIF_TIFF)
		.value("FIF_PSD", FIF_PSD)
		.value("FIF_GIF", FIF_GIF)
		.value("FIF_WEBP", FIF_WEBP)
		.export_values();

	py::enum_<FREE_IMAGE_FILTER>(m, "FREE_IMAGE_FILTER")
		.value("FILTER_BOX", FILTER_BOX)
		.value("FILTER_BICUBIC", FILTER_BICUBIC)
		.value("FILTER_BILINEAR", FILTER_BILINEAR)
		.value("FILTER_BSPLINE", FILTER_BSPLINE)
		.value("FILTER_CATMULLROM", FILTER_CATMULLROM)
		.value("FILTER_LANCZOS3", FILTER_LANCZOS3)
		.export_values();

    py::class_<Bitmap>(m, "Bitmap")
		.def(py::init<>())
		.def(py::init(&CreateFromNumpy))
        .def("width", &Bitmap::Width)
        .def("height", &Bitmap::Height)
        .def("channels", &Bitmap::Channels)
        .def("bits_per_pixel", &Bitmap::BitsPerPixel)
        .def("scan_width", &Bitmap::ScanWidth)
        .def("is_grey", &Bitmap::IsGrey)
        .def("is_rgb", &Bitmap::IsRGB)
        .def("num_bytes", &Bitmap::NumBytes)
        .def("convert_to_raw_bits", &Bitmap::ConvertToRawBits)
        .def("read", &Bitmap::Read, py::arg("path"), py::arg("as_rgb") = true)
        .def("write", &Bitmap::Write, py::arg("path"), py::arg("format") = FIF_UNKNOWN, py::arg("flags") = 0)
        .def("smooth", &Bitmap::Smooth)
        .def("rescale", &Bitmap::Rescale, py::arg("new_width"), py::arg("new_height"), py::arg("filter") = FILTER_BILINEAR)
        .def("clone", &Bitmap::Clone)
        .def("clone_as_grey", &Bitmap::CloneAsGrey)
        .def("clone_as_rgb", &Bitmap::CloneAsRGB)
		;

    py::class_<SiftExtractor>(m, "SiftExtractor")
		.def(py::init<std::shared_ptr<SiftExtractionOptions> >())
        .def("extract_sift", &SiftExtractor::ExtractSiftFeaturesGPU)
        .def("set_keypoints", &SiftExtractor::SetKeyPoints)
		;

    py::class_<FeatureKeypoint>(m, "FeatureKeypoint")
		.def(py::init<>())
		.def(py::init<float, float>(), py::arg("x"), py::arg("y"))
		.def(py::init<float, float, float, float>(), py::arg("x"), py::arg("y"), py::arg("scale"), py::arg("orientation"))
		.def(py::init<float, float, float, float, float, float>(), py::arg("x"), py::arg("y"), py::arg("a11"), py::arg("a12"), py::arg("a21"), py::arg("a22"))
        .def_readwrite("x", &FeatureKeypoint::x)
        .def_readwrite("y", &FeatureKeypoint::y)
        .def_readwrite("a11", &FeatureKeypoint::a11)
        .def_readwrite("a12", &FeatureKeypoint::a12)
        .def_readwrite("a21", &FeatureKeypoint::a21)
        .def_readwrite("a22", &FeatureKeypoint::a22)
        .def("compute_scale", &FeatureKeypoint::ComputeScale)
        .def("compute_scale_x", &FeatureKeypoint::ComputeScaleX)
        .def("compute_scale_y", &FeatureKeypoint::ComputeScaleY)
        .def("compute_orientation", &FeatureKeypoint::ComputeOrientation)
        .def("compute_shear", &FeatureKeypoint::ComputeShear)
		;
}
