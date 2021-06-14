# Research on Index Selection Problem Based on Deep Reinforcement Learning with Dynamic Workload

# What does it do?
This is an index recommendation tool to recommend an index configuration for Database under some limits (such as max index number) with dynamic workloads. Our work followed the job of HaiLan (the address of code is http://github.com/rmitbggroup/IndexAdvisor).

# What do I need to run it?
1. You should install a [PostgreSQL](https://www.postgresql.org/) database instance with [HypoPG extension](https://hypopg.readthedocs.io/en/latest/).
2. You should install the required python packages (see environment.yaml exported from conda).
3. In this code, we adopt TPC-H. Thus, you construct your own TPC-H database instance. 
4. We need the TPC-H tool to generate the workload. You can download it from this [page](http://tpc.org/tpc_documents_current_versions/current_specifications5.asp).

# How do I run it?
1. generate one workload with fixed query pattern and candidate index set for this workload by Utility/GenCandidates.py
2. generate different workloads with fixed size by Utility/GenQueryDistribution.py
3. generate the distinctions of candidate indexes by Entry/GenDistinctOfCandidates.py
4. For static workloads, you can find the entry in Entry/EntryM3DP.py
5. For dynamic workloads, you can find the entry in Entry/EntryForIndexUpdate.py

