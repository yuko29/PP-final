CXX=g++
CXXFLAGS=-Iobjs/ -I.src/ -O3 -std=c++11
OBJDIR=objs
SRC=src
EXE=main

all: $(EXE)

dirs:
	/bin/mkdir -p $(OBJDIR)/

OBJS=$(OBJDIR)/main.o $(OBJDIR)/parallel_inv.o $(OBJDIR)/lib_mat.o $(OBJDIR)/lib_mem.o $(OBJDIR)/lib_sort.o $(OBJDIR)/lib_testing_ref.o $(OBJDIR)/lib_testing.o

$(EXE): dirs $(OBJS)
	$(CXX) -o $@ $(OBJS) 

$(OBJDIR)/%.o: $(SRC)/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

.PHONY: clean

clean:
	rm -rf $(OBJDIR) $(EXE)