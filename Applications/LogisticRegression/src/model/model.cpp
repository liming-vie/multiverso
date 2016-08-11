#include "model/model.h"
#include "model/ps_model.h"

#include "util/log.h"
#include "util/common.h"
#include "updater/updater.h"
#include "multiverso/io/io.h"

namespace logreg {

template<typename EleType>
Model<EleType>::Model(Configure& config) :
  updater_(nullptr),
  computation_time_(0.0),
  compute_count_(0) {
  num_row_ = config.output_size;
  minibatch_size_ = config.minibatch_size;

  size_t size = (size_t)config.input_size * num_row_;

  // a little trick
  if (config.objective_type == "ftrl") {
    ftrl_ = true;
    table_ = (DataBlock<EleType>*)DataBlock<FTRLEntry<EleType>>
      ::GetBlock(true, size);
    table_wk_ = (DataBlock<EleType>*)DataBlock<FTRLEntry<EleType>>
      ::GetBlock(true, size);
    delta_ = (DataBlock<EleType>*)DataBlock<FTRLGradient<EleType>>
      ::GetBlock(true, size);
    delta_wk_ = (DataBlock<EleType>*)DataBlock<FTRLGradient<EleType>>
      ::GetBlock(true, size);
  } else {
    ftrl_ = false;
    table_ = DataBlock<EleType>::GetBlock(config.sparse, size);
    table_wk_ = DataBlock<EleType>::GetBlock(config.sparse, size);
    delta_ = DataBlock<EleType>::GetBlock(config.sparse, size);
    delta_wk_ = DataBlock<EleType>::GetBlock(config.sparse, size);
  }

  table_->Clear();  // will set value to zero when dense

  if (config.init_model_file != "") {
    Load(config.init_model_file);
  }

  updater_ = Updater<EleType>::Get(config);

  objective_ = Objective<EleType>::Get(config);

  Log::Write(Info, "Init local model, size [%d, %d]\n", 
    num_row_, config.input_size);
}

template<typename EleType>
Model<EleType>::~Model() {
  delete objective_;
  delete updater_;

  delete table_;
  delete delta_;
}

template<typename EleType>
DataBlock<EleType>* Model<EleType>::CreateDelta(Configure& config) {
  size_t size = (size_t)config.input_size * num_row_;
  if (config.objective_type == "ftrl") {
    return (DataBlock<EleType>*)DataBlock<FTRLGradient<EleType>>
      ::GetBlock(true, size);
  }
  else {
    DataBlock<EleType>::GetBlock(config.sparse, size);
  }
}

template<typename EleType>
inline float Model<EleType>::GetGradient(Sample<EleType>* sample,
  DataBlock<EleType>* delta, DataBlock<EleType>* model) {
  return objective_->Gradient(sample, model == nullptr ? table_ : model, delta);
}

template<typename EleType>
void Model<EleType>::AverageGradient(DataBlock<EleType>* delta, size_t batch_size) {
  if (batch_size > 1) {
    if (delta->sparse()) {
      if (ftrl_) {
        SparseBlockIter<FTRLGradient<EleType>> iter
          ((DataBlock<FTRLGradient<EleType>>*)delta);
        while (iter.Next()) {
          iter.Value()->delta_z = (EleType)(iter.Value()->delta_z
            / static_cast<double>(batch_size));
          iter.Value()->delta_n = (EleType)(iter.Value()->delta_n
            / static_cast<double>(batch_size));
        }
      }
      else {
        SparseBlockIter<EleType> iter(delta);
        while (iter.Next()) {
          (*iter.Value()) = (EleType)(*iter.Value()
            / static_cast<double>(batch_size));
        }
      }
    }
    else {
      EleType* raw = static_cast<EleType*>(delta->raw());
      for (size_t i = 0; i < delta->size(); ++i) {
        raw[i] = (EleType)(raw[i] / static_cast<double>(batch_size));
      }
    }
  }
}

template<typename EleType>
void Model<EleType>::SaveTableToTableWK(Configure &config) {
  if (config.sparse) {
    table_wk_->Clear();
  }
  table_wk_->Set(table_);
}


template<typename EleType>
void Model<EleType>::SetBatchGradient(DataBlock<EleType>* delta) {
  batch_gradient_ = delta;
}

template<typename EleType>
float Model<EleType>::Update(int count, Sample<EleType>** samples) {
  float train_loss = 0.0f;
  // process each batch
  for (int i = 0; i < count; i += minibatch_size_) {
    ++compute_count_;
    timer_.Start();
    // compute delta
    delta_->Clear(); // delta_wj
    delta_wk_->Clear();
    int upper = i + minibatch_size_;
    upper = upper > count ? count : upper;
    for (int j = i; j < upper; ++j) {
      train_loss += GetGradient(samples[j], delta_);
      GetGradient(samples[j], delta_wk_, table_wk_);
    }
    
    // TODO: only support LR
    if (delta_->sparse()) {
      EleType* val;
      {
        SparseBlockIter<EleType> iter(batch_gradient_);
        while (iter.Next()) {
          if (val = delta_->Get(iter.Key())){
            *val += *iter.Value();
          }
          else {
            delta_->Set(iter.Key(), iter.Value());
          }
        }
      }
      SparseBlockIter<EleType> iter(delta_wk_);
      while (iter.Next()) {
        if (val = delta_->Get(iter.Key())){
          *val -= *iter.Value();
        }
        else {
          delta_->Set(iter.Key(), 0 - *iter.Value());
        }
      }
    }
    else {
      EleType* raw = reinterpret_cast<EleType*>(delta_->raw());
      EleType* raw_wk = reinterpret_cast<EleType*>(delta_wk_->raw());
      EleType* raw_batch = reinterpret_cast<EleType*>(batch_gradient_->raw());
      for (size_t idx = 0; idx < delta_->size(); ++idx) {
        raw[idx] -= raw_wk[idx] - raw_batch[idx];
      }
    }
    AverageGradient(delta_, upper - i);

    computation_time_ += timer_.ElapseMilliSeconds();
    // update delta
    UpdateTable(delta_);
  }
  return train_loss;
}

template<typename EleType>
void Model<EleType>::DisplayTime() {
  if (compute_count_ == 0) {
    return;
  }
  Log::Write(Info, "average computation time: %fms\n", 
    computation_time_ / compute_count_);

  computation_time_ = 0;
  compute_count_ = 0;
}

template<typename EleType>
inline void Model<EleType>::UpdateTable(DataBlock<EleType>* delta) {
  // Log::Write(Debug, "Local model updating %d rows\n", update_idx_.size());
  timer_.Start();
  updater_->Update(table_, delta);
  computation_time_ += timer_.ElapseMilliSeconds();
}

template<typename EleType>
int Model<EleType>::Predict(int count, Sample<EleType>**samples, 
  EleType**predicts) {
  int correct(0);
  for (int i = 0; i < count; ++i) {
    this->objective_->Predict(samples[i], this->table_, predicts[i]);
    if (objective_->Correct(samples[i]->label, predicts[i])) {
      ++correct;
    }
  }
  return correct;
}

template<typename EleType>
void Model<EleType>::Load(const std::string& model_file) {
  auto stream = multiverso::StreamFactory::GetStream(
    multiverso::URI(model_file),
    multiverso::FileOpenMode::BinaryRead);
  if (table_->sparse()) {
    size_t size;
    size_t key;
    stream->Read(&size, sizeof(size_t));
    if (ftrl_) {
      FTRLEntry<EleType> val;
      for (size_t i = 0; i < size; ++i) {
        stream->Read(&key, sizeof(size_t));
        stream->Read(&val, sizeof(FTRLEntry<EleType>));
        ((DataBlock<FTRLEntry<EleType>>*)table_)->Set(key, &val);
      }
    } else {
      EleType val;
      for (size_t i = 0; i < size; ++i) {
        stream->Read(&key, sizeof(size_t));
        stream->Read(&val, sizeof(EleType));
        table_->Set(key, &val);
      }
    }
  } else {
    stream->Read(table_->raw(), table_->size() * sizeof(EleType));
  }
  delete stream;
  Log::Write(Info, "Load model from file %s\n", model_file.c_str());
}

template<typename EleType>
void Model<EleType>::Store(const std::string& model_file) {
  auto stream = multiverso::StreamFactory::GetStream(
    multiverso::URI(model_file),
    multiverso::FileOpenMode::BinaryWrite);
  if (table_->sparse()) {
    size_t tmp = table_->size();
    stream->Write(&tmp, sizeof(size_t));
    if (ftrl_) {
      SparseBlockIter<FTRLEntry<EleType>> iter(
        (DataBlock<FTRLEntry<EleType>>*)table_);
      while (iter.Next()) {
        tmp = iter.Key();
        stream->Write(&tmp, sizeof(size_t));
        stream->Write(iter.Value(), sizeof(FTRLEntry<EleType>));
      }
    } else {
      SparseBlockIter<EleType> iter(table_);
      while (iter.Next()) {
        tmp = iter.Key();
        stream->Write(&tmp, sizeof(size_t));
        stream->Write(iter.Value(), sizeof(EleType));
      }
    }
  } else {
    stream->Write(table_->raw(), table_->size() * sizeof(EleType));
  }
  delete stream;
}

template<typename EleType>
Model<EleType>* Model<EleType>::Get(Configure& config) {
  if (config.use_ps) {
    return new PSModel<EleType>(config);
  } else {
    return new Model<EleType>(config);
  }
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(Model);

} // namespace logreg
