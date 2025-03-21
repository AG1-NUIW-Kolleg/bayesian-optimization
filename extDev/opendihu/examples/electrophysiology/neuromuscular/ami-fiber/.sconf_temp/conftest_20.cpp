
    #include <iostream>
    #include <cstdlib>
    #include <fstream>
    #include <precice/SolverInterface.hpp>
    
    int main()
    {
      std::ofstream file("install_precice-config.xml");
      file << R"(<?xml version="1.0"?>
<precice-configuration>
  <solver-interface dimensions="3">
    
    <!-- Data fields that are exchanged between the solvers -->
    <data:scalar name="Data"/>

    <!-- A common mesh that uses these data fields -->
    <mesh name="Mesh">
      <use-data name="Data"/>
    </mesh>
    
    <participant name="Participant1">
      <!-- Makes the named mesh available to the participant. Mesh is provided by the solver directly. -->
      <use-mesh name="Mesh" provide="yes"/>
    </participant>
    <participant name="Participant2">
      <!-- Makes the named mesh available to the participant. Mesh is provided by the solver directly. -->
      <use-mesh name="Mesh" provide="yes"/>
    </participant>
    <m2n:sockets from="Participant1" to="Participant2" network="lo" />

    <coupling-scheme:serial-explicit>
      <participants first="Participant1" second="Participant2"/>
      <time-window-size value="0.01"/>
      <max-time value="0.05"/>
      <exchange data="Data" mesh="Mesh" from="Participant1" to="Participant2"/>
    </coupling-scheme:serial-explicit>
  </solver-interface>
</precice-configuration>
)";
      file.close();
    
      precice::SolverInterface solverInterface("Participant1","install_precice-config.xml",0,1);
      //solverInterface.initialize();
      //solverInterface.finalize();
      
      int ret = system("rm -f install_precice-config.xml");
    
      return EXIT_SUCCESS;
    }

