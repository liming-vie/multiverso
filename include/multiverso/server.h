#ifndef MULTIVERSO_SERVER_H_
#define MULTIVERSO_SERVER_H_

#include <string>
#include <vector>

#include "multiverso/actor.h"

namespace multiverso {

class ServerTable;
class Synchronizer;

class Server : public Actor {
public:
  Server();
  virtual ~Server() = default;
  static Server* GetServer();
  int RegisterTable(ServerTable* table);
protected:
  virtual void ProcessGet(MessagePtr& msg);
  virtual void ProcessAdd(MessagePtr& msg);
  // contains the parameter data structure and related handle method
  // Synchronizer* sync_;
  std::vector<ServerTable*> store_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_SERVER_H_
