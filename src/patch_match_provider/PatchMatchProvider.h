//
// Abstract superclass for patch match algorithms.
//

#ifndef PATCHMATCH_PATCHMATCHPROVIDER_H
#define PATCHMATCH_PATCHMATCHPROVIDER_H


#include "../OffsetMap.h"

class PatchMatchProvider {
public:
    virtual void match(OffsetMap *offset_map) = 0;
};
#endif //PATCHMATCH_PATCHMATCHPROVIDER_H
