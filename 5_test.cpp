/*
 * Program to run object detection on images using
 * a trained model graph. Uses Tensorflow library for
 * running graph and opencv for image manipulation
 * Author: Navdeep Dahiya; ndahiya3@gatech.edu
 */

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "labelmap.pb.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;
using tensorflow::StringPiece;

using namespace std;
using namespace cv;

// Global file paths
string test_image_dir = "./test_images";
string graph_path = "./inference_graph/frozen_inference_graph.pb";
string label_map_file = "./label_map.pbtxt";
const bool save_output_images = true;
string out_dir = "./output_images";

// Check if a given file path is jpg
bool is_jpg(string file_name);

// Check if necessary directories and files exists
bool check_input_paths_files_exist();

// Read all jpg images from input directory, list of filenames in
// out_filenames
// Return false if unsuccessful
bool read_test_jpgs(const string &in_dir, vector<string> &out_filenames);

// Convert opencv image mat to tensorflow Tensor(4D), UINT8
// Returns error code in tensorflow Status
Status read_tensor_from_cvmat(const Mat &image, Tensor &out_tensor);

// Draw an object bounding box on image with given coordinates
// Also print the score and object class label on top of object
void draw_bounding_box(Mat &image, float ymin, float xmin, float ymax,
                       float xmax, float score, const string &lbl);

// Extract box coordinates and scores of top detections (score > threshold)
// and draw each box on input image
void draw_bounding_boxes(Mat &image, tensorflow::TTypes<float>::Flat &scores,
                         tensorflow::TTypes<float>::Flat &classes,
                         tensorflow::TTypes<float, 3>::Tensor &boxes,
                         tensorflow::TTypes<float>::Flat &num_detections,
                         map<int, string> &labels_map);

// Read a google protobuffer object label text file and create object
// <class id, class name> map pairs for each possible class
// Return false in unsuccessful
bool read_label_map(string map_pbtxt, map<int, string> &labels_map);

// Reads a model graph definition from disk, and creates a session object
// used to run the graph on input images
// Returns tensorflow Status error code
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);

// Reads jpg image as tensor from input file name
// Returns tensorflow Status error code
// Not used in current code as model was trained using images
// loaded by opencv which loads them as BGR by default.
// Also tensorflow seems to have capability to draw boxes
// but not put text on images
Status read_tensor_from_image(const string &file_name,
                              vector<Tensor> *out_tensors);

// Main function to read all jpg images from input directory,
// run inference graph on each image, draw detection bounding boxes along with
// class name and scores on each image. Also optionally saves output image in
// save directory
int main() {
  cout << "starting program . . ." << endl;
  if (!check_input_paths_files_exist())
    return -1;

  // Read all jpg files from folder
  vector<string> test_file_names;
  if (!read_test_jpgs(test_image_dir, test_file_names))
    return -1;

  // Read trained graph
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Set input & output nodes names
  string input_layer = "image_tensor:0";
  vector<string> output_layer = {"detection_boxes:0", "detection_scores:0",
                                 "detection_classes:0", "num_detections:0"};

  // Get object class labels from labels_map.pbtxt
  map<int, string> labels_map;
  bool read_label_map_status = read_label_map(label_map_file, labels_map);
  if (!read_label_map_status)
    return -1;

  // Read and run each jpg image through inference graph
  for (vector<string>::iterator it = test_file_names.begin();
       it != test_file_names.end(); ++it) {

    string curr_file = *it;
    cout << "Processing file: " << curr_file << endl;

    // Read image from disk
    Mat image = imread(curr_file.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) { // Check for invalid input
      cout << "Could not open or find the image" << std::endl;
      return -1;
    }

    int height = image.rows;
    int width = image.cols;
    int ch = image.channels();

    Tensor curr_tensor(tensorflow::DT_FLOAT,
                       tensorflow::TensorShape({1, height, width, ch}));

    // Convert mat image to tensor
    Status read_tensor_status = read_tensor_from_cvmat(image, curr_tensor);
    if (!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return -1;
    }

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status =
        session->Run({{input_layer, curr_tensor}}, output_layer, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }

    // Get results from outputs tensor
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    tensorflow::TTypes<float, 3>::Tensor boxes =
        outputs[0].flat_outer_dims<float, 3>();

    // Draw bounding boxes with class labels for detections
    draw_bounding_boxes(image, scores, classes, boxes, num_detections,
                        labels_map);

    // Optionally save image
    if (save_output_images) {
      int found = curr_file.find_last_of("/");
      string file_name = curr_file.substr(found + 1);
      string file_path = out_dir + "/" + file_name;
      imwrite(file_path.c_str(), image);
    }

    // Display image
    namedWindow("Detections", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Detections", image);                // Show our image inside it.

    waitKey(0);
  }
  return 0;
}

// Check if a given file path is jpg
bool is_jpg(string file_name) {
  string suffix = ".jpg";
  if (file_name.size() > suffix.size() &&
      file_name.compare(file_name.size() - suffix.size(), suffix.size(),
                        suffix) == 0)
    return true;
  else
    return false;
}

bool check_input_paths_files_exist() {
  // Check test images directory
  struct stat sb;
  if (stat(test_image_dir.c_str(), &sb) != 0) {
    cout << "Input images directory path doesn't exist." << endl;
    return false;
  } else {
    if (!S_ISDIR(sb.st_mode)) {
      cout << "Input images directory path is not a directory." << endl;
      return false;
    }
  }

  // Check if at least one jpg file in test images directory
  DIR *dp;
  struct dirent *dirp;
  dp = opendir(test_image_dir.c_str());
  if (dp == NULL) {
    cout << "Error reading test images directory." << endl;
    return false;
  }
  bool found_jpeg = false;
  while ((dirp = readdir(dp)) != NULL)
    if (is_jpg(string(dirp->d_name))) {
      found_jpeg = true;
      break;
    }
  closedir(dp);
  if (!found_jpeg) {
    cout << "No jpg files in input images directory." << endl;
    return false;
  }

  // Check if frozen inference graph exists
  if (stat(graph_path.c_str(), &sb) != 0) {
    cout << "Inference graph path doesn't exist." << endl;
    return false;
  } else {
    if (!S_ISREG(sb.st_mode)) {
      cout << "Inference graph path is not a valid file." << endl;
      return false;
    }
  }

  // Check if label map file exists
  if (stat(label_map_file.c_str(), &sb) != 0) {
    cout << "Label map file doesn't exist." << endl;
    return false;
  } else {
    if (!S_ISREG(sb.st_mode)) {
      cout << "Label map file is not a valid file." << endl;
      return false;
    }
  }

  // Check if output images directory exists
  if (save_output_images) {
    if (stat(out_dir.c_str(), &sb) != 0) {
      cout << "Output images directory path doesn't exist." << endl;
      return false;
    } else {
      if (!S_ISDIR(sb.st_mode)) {
        cout << "Output images directory path is not a directory." << endl;
        return false;
      }
    }
  }

  // All checks passed
  return true;
}

bool read_test_jpgs(const string &in_dir, vector<string> &out_filenames) {

  DIR *dp;
  struct dirent *dirp;

  dp = opendir(in_dir.c_str());
  if (dp == NULL) {
    cout << "Error reading the data directory.\n";
    return false;
  }

  while ((dirp = readdir(dp)) != NULL)
    if (is_jpg(string(dirp->d_name))) // Parse filenames ending in .jpg only
      out_filenames.push_back(in_dir + "/" + string(dirp->d_name));
  closedir(dp);

  if (out_filenames.size() == 0) {
    cout << "No jpg files found in directory " << in_dir << endl;
    return false;
  }

  return true;
}

Status read_tensor_from_cvmat(const Mat &image, Tensor &out_tensor) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;

  // Use a mat image with shallow copy (pointer) to tensor data
  // Copy the data from read input mat image to the pointer image
  // which effectively copies the data from image to tensor
  // https://github.com/tensorflow/tensorflow/issues/8033
  float *data_ptr = out_tensor.flat<float>().data();
  Mat ptr_image(image.rows, image.cols, CV_32FC3, data_ptr);
  image.convertTo(ptr_image, CV_32FC3);

  auto input_tensor =
      Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
  vector<pair<string, Tensor>> inputs = {{"input", out_tensor}};

  auto uint8_caster =
      Cast(root.WithOpName("uint8_caster"), out_tensor, tensorflow::DT_UINT8);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  vector<Tensor> out_tensors;
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({inputs}, {"uint8_caster"}, {}, &out_tensors));

  out_tensor = out_tensors[0];

  return Status::OK();
}

void draw_bounding_box(Mat &image, float ymin, float xmin, float ymax,
                       float xmax, float score, const string &lbl) {
  // Draw a bounding box in green color on given image with input
  // coordinates which are normalized
  int h = image.rows;
  int w = image.cols;
  Scalar green_color(0, 255, 0);
  Scalar black_color(0, 0, 0);

  // Bounding boxes are normalized. Convert to absolute coordinates
  Point top_left((int)(xmin * w), (int)(ymin * h));
  Point bot_right((int)(xmax * w), (int)(ymax * h));

  // Draw bounding box
  rectangle(image, top_left, bot_right, green_color, 4);

  // Add class label with score
  int score_perc = (int)(score * 100);
  string cls_lbl = lbl + ": " + to_string(score_perc) + "%";

  int font_face = FONT_HERSHEY_SIMPLEX;
  double font_scale = 0.6;
  int thickness = 1;

  int baseline = 0;
  Size text_size =
      getTextSize(cls_lbl, font_face, font_scale, thickness, &baseline);

  // Text bounding box and text origin coordinates
  Point cls_tl(top_left.x, top_left.y - text_size.height);
  Point cls_br(top_left.x + text_size.width, top_left.y);
  Point txt_orig(cls_tl.x, cls_br.y);

  // Draw text bounding box and put class label and score
  rectangle(image, cls_tl, cls_br, green_color, -1);
  putText(image, cls_lbl, txt_orig, font_face, font_scale, black_color,
          thickness, CV_AA);
}

void draw_bounding_boxes(Mat &image, tensorflow::TTypes<float>::Flat &scores,
                         tensorflow::TTypes<float>::Flat &classes,
                         tensorflow::TTypes<float, 3>::Tensor &boxes,
                         tensorflow::TTypes<float>::Flat &num_detections,
                         map<int, string> &labels_map) {
  // Draw bounding boxes around detected objects
  // Python vis_util.visualize_boxes_and_labels_on_image_array
  // uses a default score threshold of 0.5 to select a box to draw
  float threshold = 0.5;
  for (int i = 0; i < (int)num_detections(0); i++) {
    if (scores(i) > threshold) {
      float ymin = boxes(0, i, 0);
      float xmin = boxes(0, i, 1);
      float ymax = boxes(0, i, 2);
      float xmax = boxes(0, i, 3);
      string label = labels_map[int(classes(i))];
      draw_bounding_box(image, ymin, xmin, ymax, xmax, scores(i), label);
    }
  }
}

bool read_label_map(string map_pbtxt, map<int, string> &labels_map) {
  // Read a text lable_map.pbtxt file which could have multiple object
  // categories. Create <key=id, value=name> map for each class

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // Open label_map.pbtxt
  int file_descriptor = open(map_pbtxt.c_str(), O_RDONLY); // C system call
  if (file_descriptor < 0) {
    cerr << "Label map file not found or could not be opened." << endl;
    return false;
  }
  google::protobuf::io::FileInputStream file_in(file_descriptor);
  file_in.SetCloseOnDelete(true);

  // Parse all object classes
  labelmap::ObjectClasses all_classes;
  if (!google::protobuf::TextFormat::Parse(&file_in, &all_classes)) {
    cerr << "Failed to parse label map." << endl;
    return false;
  }

  // Iterate through all labels and create map
  for (int i = 0; i < all_classes.item_size(); i++) {
    const labelmap::Category curr_label = all_classes.item(i);
    int id = curr_label.id();
    string name = curr_label.name();
    labels_map.insert(pair<int, string>(id, name));
  }

  // Test output
  cout << "Class labels found ..." << endl;
  for (map<int, string>::iterator it = labels_map.begin();
       it != labels_map.end(); ++it)
    cout << it->first << " ==> " << it->second << endl;
  return true;
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

Status read_tensor_from_image(const string &file_name,
                              vector<Tensor> *out_tensors) {

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;

  string input_name = "file_reader";
  string original_name = "identity";
  string output_name = "normalized";

  auto file_reader =
      tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);

  // We know its a jpg so decode as jpg
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                            DecodeJpeg::Channels(wanted_channels));

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  auto original_image = Identity(root.WithOpName(original_name), image_reader);

  // Cast to uint8
  auto uint8_caster = Cast(root.WithOpName("uint8_caster"), original_image,
                           tensorflow::DT_UINT8);
  // The convention for image ops in TensorFlow is that all images are
  // expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander =
      ExpandDims(root.WithOpName(output_name), uint8_caster, 0);

  // This runs the GraphDef network definition that we've just constructed,
  // and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({}, {output_name, original_name}, {}, out_tensors));
  return Status::OK();
}
