//
// Abstract superclass for patch match algorithms.
//

#ifndef PATCHMATCH_PATCHMATCHPROVIDER_H
#define PATCHMATCH_PATCHMATCHPROVIDER_H


#include <memory>
#include "../OffsetMap.h"

class PatchMatchProvider {
public:
    virtual std::shared_ptr<OffsetMap> match() = 0;
};
#endif //PATCHMATCH_PATCHMATCHPROVIDER_H
