# Default options.
CXX := g++
CXX_WARNING_OPTIONS := -Wall -Wextra -Wno-unused-result
CXXFLAGS := -std=c++11 -fPIC $(CXX_WARNING_OPTIONS)
SRC_DIR := src/tools
BUILD_DIR := build
LIB_DIR := lib
PYLIB_DIR := /home/cyrus/anaconda2/pkgs/python-3.7.4-h265db76_1/include/python3.7m/
MODULE := wf
EXE := csf_gen

# Libraries.
CXXFLAGS := $(CXXFLAGS) -I $(LIB_DIR) -I $(PYLIB_DIR)

# Sources and intermediate objects.
SRCS := $(shell find $(SRC_DIR) ! -name $(EXE).cc ! -name $(MODULE).cc -name "*.cc")
MOD_SRCS := $(shell find $(SRC_DIR) -name $(MODULE).cc)
EXE_SRCS := $(shell find $(SRC_DIR) -name $(EXE).cc)
HEADERS := $(shell find $(SRC_DIR) -name "*.h")
SUBMODULES := $(LIB_DIR)/hps
OBJS := $(SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
MOD_OBJS := $(MOD_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
EXE_OBJS := $(EXE_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)

.SUFFIXES:

all: $(MODULE) $(EXE)

clean:
	rm -rf $(BUILD_DIR)
	rm $(LIB_DIR)/$(MODULE).so
	rm $(LIB_DIR)/$(EXE)

$(MODULE): $(OBJS) $(MOD_OBJS) $(HEADERS) 
	echo $(MOD_SRCS)
	echo $(MOD_OBJS)
	$(CXX) -shared $(MOD_OBJS) $(OBJS) -o $(LIB_DIR)/$(MODULE).so

$(EXE): $(OBJS) $(EXE_OBJS) $(HEADERS)
	$(CXX) $(EXE_OBJS) $(OBJS) -o $(LIB_DIR)/$(EXE)

$(OBJS) $(MOD_OBJS) $(EXE_OBJS): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(HEADERS)
	mkdir -p $(@D) && $(CXX) $(CXXFLAGS) -c $< -o $@
