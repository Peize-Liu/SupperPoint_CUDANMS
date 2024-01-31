#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <iostream>
#include <fstream>

#include "trt_buffers.h"

const std::string engine_path = "/root/sp_ws/sp_ws/superpoint_v1_dyn_size_onnx_152_304.trt";
std::vector<std::string> input_tensor_name = {"image"};
std::vector<std::string> ouput_tensor_name = {"semi", "desc"};

class SuperPointLogger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kWARNING){
      std::cout << msg <<std::endl;
    }
  }
};

void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, double threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
            // printf("keypoint: %d, %d, score: %f\n", location[0], location[1], scores[i]);
        }
    }
    scores.swap(new_scores);
}

class SpExcutor{
	public:
		SpExcutor(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr,
		 const std::vector<std::string>& input_tensor_name, const std::vector<std::string>& ouput_tensor_name): 
		engine_ptr_(engine_ptr), input_tensor_name_(input_tensor_name), ouput_tensor_name_(ouput_tensor_name){
			if (!engine_ptr_) {
				std::cerr << "Failed to create TensorRT Engine." << std::endl;
			}
			nv_context_ptr_ = engine_ptr_->createExecutionContext();
			if (!nv_context_ptr_) {
				std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
			}
			nv_context_ptr_->setBindingDimensions(0, nvinfer1::Dims4(1, 1, 150, 300));
			cudaStreamCreate(&stream_);
			buffer_manager_ptr_ = std::make_unique<CudaMemoryManager::BufferManager>(engine_ptr_, 1, nv_context_ptr_);
			if (!buffer_manager_ptr_) {
				std::cerr << "Failed to create TensorRT Buffer Manager." << std::endl;
			}
			
		}
		~SpExcutor(){
			if (nv_context_ptr_) {
				nv_context_ptr_->destroy();
			}
			// if (buffer_manager_ptr_) {
			// 	buffer_manager_ptr_->clear();
			// }
			if (stream_) {
				cudaStreamDestroy(stream_);
			}
		}
		void print_info(){
			printf("input_tensor_name: %s index %d \n", input_tensor_name_[0].c_str(), this->engine_ptr_->getBindingIndex(input_tensor_name_[0].c_str()));
			//get input dimension 
			nvinfer1::Dims input_dims = this->engine_ptr_->getBindingDimensions(this->engine_ptr_->getBindingIndex(input_tensor_name_[0].c_str()));
			printf("input_dims: %d %d %d %d %d\n", input_dims.nbDims, input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3]);
			//get output dimension 0 
			nvinfer1::Dims output_dims = this->engine_ptr_->getBindingDimensions(this->engine_ptr_->getBindingIndex(ouput_tensor_name_[0].c_str()));
			printf("output_dims: %d %d %d %d %d\n", output_dims.nbDims, output_dims.d[0], output_dims.d[1], output_dims.d[2], output_dims.d[3]);
			//get ouput dimension 1
			nvinfer1::Dims output_dims1 = this->engine_ptr_->getBindingDimensions(this->engine_ptr_->getBindingIndex(ouput_tensor_name_[1].c_str()));
			printf("output_dims: %d %d %d %d %d\n", output_dims1.nbDims, output_dims1.d[0], output_dims1.d[1], output_dims1.d[2], output_dims1.d[3]);
			//set output dimension 0

		}
		void inference(const cv::Mat& image, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc){
			//get input dimension
			printf("inference\n");
			void* input_buffer =  buffer_manager_ptr_->getHostBuffer(input_tensor_name_[0]);
			if (!input_buffer) {
				std::cerr << "Failed to get input buffer." << std::endl;
			}
			//copy image to input buffer
			cv::Mat image_float;
			image.convertTo(image_float, CV_32FC1, 1/255.0);
			cv::imshow ("image", image_float);
			cv::waitKey(0);
			std::memcpy(input_buffer, image_float.data, 1*150*300* sizeof(float));
	
			//execute inference
			buffer_manager_ptr_->copyInputToDeviceAsync();

			if (!nv_context_ptr_->enqueueV2(buffer_manager_ptr_->getDeviceBindings().data(), stream_, nullptr)) {
				std::cerr << "Failed to execute TensorRT inference." << std::endl;
			}
			//
			buffer_manager_ptr_->copyOutputToHostAsync();
			cudaStreamSynchronize(stream_);
			// memset (buffer_manager_ptr_->getHostBuffer(ouput_tensor_name_[0]) , -1.0f, 150*300);
			//get output dimension
			float* output_buffer =  static_cast<float *>( buffer_manager_ptr_->getHostBuffer(ouput_tensor_name_[0]));
			if (!output_buffer) {
				std::cerr << "Failed to get output buffer." << std::endl;
			}
			printf("output_buffer: %p\n", output_buffer);
			//copy output to cv::Mat
			// cv::Mat semi(150, 300, CV_32FC1, output_buffer);
			// // memset(output_buffer, -1.0f, 150*300);
			// cv::imshow ("semi", semi);
			// cv::waitKey(0);
			std::vector<float> scores_vec(output_buffer, output_buffer+150*300);
			std::vector<std::vector<int>> keypoints;
			find_high_score_index(scores_vec, keypoints, 150, 300, 0.015);
			printf("stop\n");
		}
	private:
		std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr_ = nullptr;
		nvinfer1::IExecutionContext* nv_context_ptr_ = nullptr;
		std::unique_ptr<CudaMemoryManager::BufferManager> buffer_manager_ptr_;
		cudaStream_t stream_;
		std::vector<std::string> input_tensor_name_;
		std::vector<std::string> ouput_tensor_name_;
};

int main(int, char**){
//init engine
	SuperPointLogger gLogger;
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (!runtime) {
		std::cerr << "Failed to create TensorRT Runtime object." << std::endl;
		return -1;
	}
  std::ifstream engine_file(engine_path, std::ios::binary);
	if (!engine_file) {
		std::cerr << "Failed to open engine file." << std::endl;
		return -1;
	}
	std::stringstream engine_buffer;
	engine_buffer << engine_file.rdbuf();
	std::string plan = engine_buffer.str();
	printf("plan size: %d\n", plan.size());
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size(), nullptr);
	if (!engine) {
		std::cerr << "Failed to create TensorRT Engine." << std::endl;
		return -1;
	}
	//create engine
	std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr(engine, InferDeleter());
	if (!nv_engine_ptr) {
		std::cerr << "Failed to create TensorRT Engine." << std::endl;
		return -1;
	}

	SpExcutor sp_excutor(nv_engine_ptr, input_tensor_name, ouput_tensor_name);
	sp_excutor.print_info();
	cv::Mat image = cv::imread("/root/sp_ws/sp_ws/example_data/undistort_images/0/2.png");
	printf("image type %d\n", image.type());
	std::vector<cv::KeyPoint> kpts;
	cv::Mat desc;
	sp_excutor.inference(image, kpts, desc);
}
