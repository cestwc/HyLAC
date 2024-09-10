OBJDIR=./o
EXEDIR=.

# Make OBJDIR if it doesn't exist
$(shell mkdir -p $(OBJDIR))

all: LAP_gurobi.o
	g++ -g $(OBJDIR)/LAP_gurobi.o -I/opt/gurobi1100/linux64/include/ -L/opt/gurobi1100/linux64/lib -lgurobi_c++ -lgurobi110 -o $(EXEDIR)/LAP_gurobi

LAP_gurobi.o: LAP_gurobi.cpp
	g++ -g -c LAP_gurobi.cpp -o $(OBJDIR)/LAP_gurobi.o -I/opt/gurobi1100/linux64/include/ -L/opt/gurobi1100/linux64/lib -lgurobi_c++ -lgurobi110

clean:
	rm -rf $(EXEDIR)/LAP_gurobi
	rm -rf $(OBJDIR)/*.o
