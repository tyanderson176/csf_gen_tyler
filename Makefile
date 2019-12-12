# Default options.
CXX := g++
CXX_WARNING_OPTIONS := -Wall -Wextra -Wno-unused-result
CXXFLAGS := -std=c++11 -fPIC -O3 $(CXX_WARNING_OPTIONS)
SRC_DIR := src/tools
BUILD_DIR := build
LIB_DIR := lib
PYLIB_DIR := /home/cyrus/anaconda2/pkgs/python-3.7.4-h265db76_1/include/python3.7m/
WF_LOAD := load_wf
CSF := spin_csf_gen
REL_PAR := rel_parity

# Libraries.
CXXFLAGS := $(CXXFLAGS) -I $(LIB_DIR) -I $(PYLIB_DIR)

# Sources and intermediate objects.
SRCS := $(shell find $(SRC_DIR) \
				! -name $(WF_LOAD).cc ! -name $(CSF).cc ! -name $(REL_PAR).cc \
				-name "*.cc")
OBJS := $(SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)

WF_LOAD_SRCS := $(shell find $(SRC_DIR) -name $(WF_LOAD).cc)
WF_LOAD_OBJS := $(WF_LOAD_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)

CSF_SRCS := $(shell find $(SRC_DIR) -name $(CSF).cc)
CSF_OBJS := $(CSF_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)

REL_PAR_SRCS := $(shell find $(SRC_DIR) -name $(REL_PAR).cc)
REL_PAR_OBJS := $(REL_PAR_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)

HEADERS := $(shell find $(SRC_DIR) -name "*.h")
SUBMODULES := $(LIB_DIR)/hps

.SUFFIXES:

all: $(CSF) $(WF_LOAD) $(REL_PAR)

clean:
	rm -rf $(BUILD_DIR)
	rm $(LIB_DIR)/$(MODULE).so
	rm $(LIB_DIR)/$(EXE)

$(WF_LOAD): $(OBJS) $(WF_LOAD_OBJS) $(HEADERS) 
	echo $(WF_LOAD)
	echo $(WF_LOAD_OBJS)
	$(CXX) -shared $(WF_LOAD_OBJS) $(OBJS) -o $(LIB_DIR)/$(WF_LOAD).so

$(REL_PAR): $(REL_PAR_OBJS)
	echo $(REL_PAR)
	echo $(REL_PAR_OBJS)
	$(CXX) -shared $(REL_PAR_OBJS) -o $(LIB_DIR)/$(REL_PAR).so

$(CSF): $(OBJS) $(CSF_OBJS) $(HEADERS)
	$(CXX) $(CSF_OBJS) $(OBJS) -o $(LIB_DIR)/$(CSF)

$(OBJS) $(CSF_OBJS) $(WF_LOAD_OBJS) $(REL_PAR_OBJS): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(HEADERS)
	mkdir -p $(@D) && $(CXX) $(CXXFLAGS) -c $< -o $@
