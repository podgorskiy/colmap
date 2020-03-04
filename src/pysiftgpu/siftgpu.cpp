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

std::shared_ptr<SiftExtractionOptions> make_extreme_options()
{
	OptionManager options;
	options.ModifyForExtremeQuality();
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


// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
static FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}


FeatureDescriptors ComputeDescriptors(const SiftExtractionOptions& options, VlCovDet* covdet, const FeatureKeypoints& keypoints);

std::pair<FeatureDescriptors, FeatureKeypoints>  ExtractCovariantSiftFeatures(const std::shared_ptr<SiftExtractionOptions>& _options,
                                     const Bitmap& bitmap)
{
	SiftExtractionOptions options = *_options;
	CHECK(options.Check());
	CHECK(bitmap.IsGrey());

	CHECK(!options.darkness_adaptivity);

	// Setup covariant SIFT detector.
	std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
			vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
	if (!covdet)
	{
		return std::pair<FeatureDescriptors, FeatureKeypoints>();
	}

	const int kMaxOctaveResolution = 1000;
	CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

	vl_covdet_set_first_octave(covdet.get(), options.first_octave);
	vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
	vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
	vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

	{
		const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
		std::vector<float> data_float(data_uint8.size());
		for (size_t i = 0; i < data_uint8.size(); ++i)
		{
			data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
		}
		vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
		                    bitmap.Height());
	}

	vl_covdet_detect(covdet.get(), options.max_num_features);

	if (!options.upright)
	{
		if (options.estimate_affine_shape)
		{
			vl_covdet_extract_affine_shape(covdet.get());
		}
		else
		{
			vl_covdet_extract_orientations(covdet.get());
		}
	}

	const int num_features = vl_covdet_get_num_features(covdet.get());
	VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

	// Sort features according to detected octave and scale.
	std::sort(
			features, features + num_features,
			[](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2)
			{
				if (feature1.o == feature2.o)
				{
					return feature1.s > feature2.s;
				}
				else
				{
					return feature1.o > feature2.o;
				}
			});

	const size_t max_num_features = static_cast<size_t>(options.max_num_features);

	FeatureKeypoints keypoints;
	// Copy detected keypoints and clamp when maximum number of features reached.
	int prev_octave_scale_idx = std::numeric_limits<int>::max();
	for (int i = 0; i < num_features; ++i)
	{
		FeatureKeypoint keypoint;
		keypoint.x = features[i].frame.x + 0.5;
		keypoint.y = features[i].frame.y + 0.5;
		keypoint.a11 = features[i].frame.a11;
		keypoint.a12 = features[i].frame.a12;
		keypoint.a21 = features[i].frame.a21;
		keypoint.a22 = features[i].frame.a22;
		keypoints.push_back(keypoint);

		const int octave_scale_idx =
				features[i].o * kMaxOctaveResolution + features[i].s;
		CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

		if (octave_scale_idx != prev_octave_scale_idx &&
		    keypoints.size() >= max_num_features)
		{
			break;
		}

		prev_octave_scale_idx = octave_scale_idx;
	}

	return std::make_pair(ComputeDescriptors(options, covdet.get(), keypoints), keypoints);
}


FeatureDescriptors ExtractCovariantSiftFeaturesGivenKeypoints(const std::shared_ptr<SiftExtractionOptions>& _options,
                                     const Bitmap& bitmap, const FeatureKeypoints& keypoints)
{
	SiftExtractionOptions options = *_options;
	CHECK(options.Check());
	CHECK(bitmap.IsGrey());

	CHECK(!options.darkness_adaptivity);

	// Setup covariant SIFT detector.
	std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
			vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
	if (!covdet)
	{
		return FeatureDescriptors();
	}

	const int kMaxOctaveResolution = 1000;
	CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

	vl_covdet_set_first_octave(covdet.get(), options.first_octave);
	vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
	vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
	vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

	{
		const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
		std::vector<float> data_float(data_uint8.size());
		for (size_t i = 0; i < data_uint8.size(); ++i)
		{
			data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
		}
		vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
		                    bitmap.Height());
	}

	vl_covdet_detect(covdet.get(), options.max_num_features);

	if (!options.upright)
	{
		if (options.estimate_affine_shape)
		{
			vl_covdet_extract_affine_shape(covdet.get());
		}
		else
		{
			vl_covdet_extract_orientations(covdet.get());
		}
	}
	return ComputeDescriptors(options, covdet.get(), keypoints);
}


FeatureDescriptors ComputeDescriptors(const SiftExtractionOptions& options, VlCovDet* covdet, const FeatureKeypoints& keypoints)
{
	FeatureDescriptors descriptors;
	{
		descriptors.resize(keypoints.size(), 128);

		const size_t kPatchResolution = 15;
		const size_t kPatchSide = 2 * kPatchResolution + 1;
		const double kPatchRelativeExtent = 7.5;
		const double kPatchRelativeSmoothing = 1;
		const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
		const double kSigma =
				kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

		std::vector<float> patch(kPatchSide * kPatchSide);
		std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

		float dsp_min_scale = 1;
		float dsp_scale_step = 0;
		int dsp_num_scales = 1;
		if (options.domain_size_pooling)
		{
			dsp_min_scale = options.dsp_min_scale;
			dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
			                 options.dsp_num_scales;
			dsp_num_scales = options.dsp_num_scales;
		}

		Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
				scaled_descriptors(dsp_num_scales, 128);

		std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
				vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
		if (!sift)
		{
			return FeatureDescriptors();
		}

		vl_sift_set_magnif(sift.get(), 3.0);

		for (size_t i = 0; i < keypoints.size(); ++i)
		{
			for (int s = 0; s < dsp_num_scales; ++s)
			{
				const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

				// VlFrameOrientedEllipse scaled_frame = features[i].frame;
				VlFrameOrientedEllipse scaled_frame;
				scaled_frame.x = keypoints[i].x - 0.5;
				scaled_frame.y = keypoints[i].y - 0.5;

				scaled_frame.a11 = keypoints[i].a11;
				scaled_frame.a12 = keypoints[i].a12;
				scaled_frame.a21 = keypoints[i].a21;
				scaled_frame.a22 = keypoints[i].a22;

				scaled_frame.a11 *= dsp_scale;
				scaled_frame.a12 *= dsp_scale;
				scaled_frame.a21 *= dsp_scale;
				scaled_frame.a22 *= dsp_scale;

				vl_covdet_extract_patch_for_frame(
						covdet, patch.data(), kPatchResolution, kPatchRelativeExtent,
						kPatchRelativeSmoothing, scaled_frame);

				vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
				                      2 * kPatchSide, patch.data(), kPatchSide,
				                      kPatchSide, kPatchSide);

				vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
				                            scaled_descriptors.row(s).data(),
				                            kPatchSide, kPatchSide, kPatchResolution,
				                            kPatchResolution, kSigma, 0);
			}

			Eigen::Matrix<float, 1, 128> descriptor;
			if (options.domain_size_pooling)
			{
				descriptor = scaled_descriptors.colwise().mean();
			}
			else
			{
				descriptor = scaled_descriptors;
			}

			if (options.normalization == SiftExtractionOptions::Normalization::L2)
			{
				descriptor = L2NormalizeFeatureDescriptors(descriptor);
			}
			else if (options.normalization ==
			         SiftExtractionOptions::Normalization::L1_ROOT)
			{
				descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
			}
			else
			{
				LOG(FATAL) << "Normalization type not supported";
			}

			descriptors.row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
		}

		descriptors = TransformVLFeatToUBCFeatureDescriptors(descriptors);
	}
	return descriptors;
}



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

    m.def("make_extreme_options", &make_extreme_options, "Returns default quality settings for feature extraction");

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

    m.def("extract_covariant_sift", &ExtractCovariantSiftFeatures);
    m.def("extract_covariant_sift_given_keypoints", &ExtractCovariantSiftFeaturesGivenKeypoints);
    m.def("resize", &Resize);
}
