#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/workspace.h"

namespace GoodBot
{

class NetOp
{
public:
NetOp(const caffe2::OperatorDef& operatorDef, const std::vector<std::string>& activeModes, bool trainable = false);
NetOp(const NetOp& op);

const std::string& GetName() const;
const caffe2::OperatorDef& GetOperatorDef() const;
const std::vector<std::string>& GetActiveModes() const;
bool IsTrainable() const;

private:
caffe2::OperatorDef OperatorDef;
std::vector<std::string> ActiveModes;
bool Trainable;
};

caffe2::TensorCPU& GetMutableTensor(const std::string& name, caffe2::Workspace& workspace);
caffe2::TensorCPU GetTensor(const caffe2::Blob &blob);
int64_t GetNumInput(const NetOp& op);
int64_t GetNumOutput(const NetOp& op);
bool HasOutputWithoutInput(const NetOp& op);
bool HasInputBlob(const std::string& blobName, const NetOp& op);
bool HasOutputBlob(const std::string& blobName, const NetOp& op); 
bool CreatesBlob(const std::string& blobName, const NetOp& op);
bool IsType(const std::string& blobName, const NetOp& op);
const std::string& GetOutputName(const NetOp& op, int64_t index);
const std::string& GetInputName(const NetOp& op, int64_t index);
bool MakesGradients(const NetOp& op);

//Returns null if the OperatorDef doesn't have the given argument.
const caffe2::Argument* GetArgument(const std::string& argumentName, const caffe2::OperatorDef& opDef);







}
