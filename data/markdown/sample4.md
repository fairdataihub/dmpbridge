

<!-- Page 1 -->

Data and Computing Resource Management Plan
1. Introduction
The goal of this Data and Computing Resources Management Plan is to provide a systematic,
centralized,andexpertlysupportedapproachtomeetingthedatamanagementandcyberinfrastruc-
tureneedsoftheOdor2Actionnetwork.Foraprojectofthissize,complexity,andinterdisciplinarity,
signi fi cant infrastructure and expertise are required to enable robust large-scale centralized data
storage,seamlessaccesstodataandcomputationalresources,andopensciencepracticeslikepublic
access to well-curated data, code, and software in established open repositories. This proposed plan
adds value by providing a means for secure and e ffi cient transfer of large datasets between theo-
166
reticians and experimentalists. As noted in the recent NSF Dear Colleague Letter NSF 19-069
“E ff ective Practices for Data,” open science practices have myriad bene fi ts for the scienti fi c and
broadercommunity;however,makingdataopenlyaccessibleandactuallyusefulviaexpertcuration
requires personnel e ff ort and technical resources. This plan leverages existing and past NSF invest-
ments to ensure that data are easily and safely stored in centralized infrastructure with simple and
fast access to data and analysis tools by the large and geographically distributed project team. In
addition, the plan enables openly available and reusable data in a sustainable way over the long
term, according to emerging best practices. These practices include the use of persistent identi fi ers
such as DataCite Digital Object Identi fi ers (DOIs), a machine-readable data management plan as
described in NSF document 19-069 referenced above, and adherence to the broader FAIR data
215,258
principles and emerging software citation principles. This plan describes how the variety and
volume of data produced will be supported with appropriate expertise and resources for the entire
lifecycle from data storage and analysis to open data sharing and long-term preservation.
2. Cyberinfrastructure Team
Digital data and computing resources will be managed by the Center for Research Data and
Digital Scholarship (CRDDS) at the University of Colorado Boulder. CRDDS brings together na-
tional experts in cyberinfrastructure and data management to support research data, computation,
and analysis across disciplines in accordance with open science principles.
Senior Personnel Thomas Hauser: Hauser is CRDDS executive director and director of re-
search computing at CU Boulder. He will oversee cyberinfrastructure and guide the use of the
computing resources for the project. Hauser is PI on the NSF Hybrid Cloud award, the NSF Cy-
berteam award and is overseeing the operations and support of the central computing and data
resources for the university. His group operates an advanced computational and data-intensive
infrastructure consisting of supercomputers, high-throughput computing and data resources con-
nected through a Science DMZ to other regional and national resources. CRDDS o ff ers a large
number of training programs for researchers and support for data-intensive research, and Research
Computing provides user support for the compute and storage systems.
SeniorPersonnelAndrewJohnson: JohnsonisCRDDSdirectorofresearchdatamanagement
and associate professor at CU Boulder. He will oversee data management, data archiving, and
data sharing/publishing for the project. Johnson’s team supports data management best practices
accordingtonationalandinternationalstandardsandrunstheCUBoulderrepository,CUScholar,
which provides infrastructure for data publishing and archiving in accordance with open science
principles. Johnson has a strong record of publications and grants, including from the Institute of
Museum and Library Services, in the area of data management support.
3. Cyberinfrastructure System Description
Hauser and Johnson will develop and support a system of interconnected resources to ensure
that data move as easy as possible from instruments and other data collection sources to a cen-
tralized storage system (PetaLibrary), where it can be accessed by the project team for analysis
and computation (Fig. 19). The data on the PetaLibrary can be accessed by researchers through
49


<!-- Page 2 -->

130 106
web-basedinterfaceslikeJupyternotebooks ortheOpenOnDemandportal. Thedataanalysis
will be performed on several computing resources. The project has access to the fi ve compute nodes
budgetedforpurchaseintheBlancacluster.Theseresourcesarededicatedtotheproject,butother
idle compute cycles of the system can also be utilized by this project through preemptible queues.
Additionally, the project will have access to the RMACC Summit supercomputer (funded by an
NSF MRI award) for computing needs that go beyond the nodes of the Blanca cluster. Hybrid
cloud computing resources will also be available to the project through the NSF CC*-funded hy-
brid cloud infrastructure that will be operational in the spring of 2020. In addition, data stored in
the PetaLibrary can be shared easily with collaborators or with anyone who requests access, even
to large data sets, using the Globus data transfer service. Data sets stored in the PetaLibrary that
are associated with publications or of potential reuse by the community can be more easily curated
for open science publication and long-term archiving in repositories supported by the team (CU
Scholar) as well as external repositories (Zenodo, OdorMapDB).
Figure19: SchematicoftheOdor2Actioncyberinfrastructuresystemfordatasharing,computing,andopenscience.
NSF Logos denote leveraging of previous NSF cyberinfrastructure investments.
Types of data: Allbehavioral,imaging,and in vivo neurophysiologicaldatawillbeorganizedby
trial/session/animal. Behavioral data includes video data and automated measurements character-
izing the behavior such as velocity, position, and sensor/head motion. Imaging data trials consist of
image stacks collected at 30 frames per second and digitized at 16-bit precision. Trials also include
time series (50 samples/s) of miniPID olfactometer odor concentrations, licking and sni ffi ng be-
havior (for mice), and metadata (odor, concentration, plume ID, other experimental information).
In vitro and other physiological data will include time series of voltage and/or current as well as
metadataoncelllocationandmorphology.Otherdatasetswillconsistofodorplumemeasurements,
robot trajectory data, simulation results, Illumina sequence data, and video data. Code for simu-
lation and analysis of the experimental data will be in Python, R, or MatLab. Data is expected to
total approximately 150 terabytes in size, necessitating the use of a large-scale storage solution and
specialized tools and work fl ows for data sharing, publishing, and archiving as described below.
Data standards: Voltagedata,images,andotherlargedatasetswillbekeptinrawformatalong
withaneasilyreadable fi lethatcontainsinformationaboutthetypeofdata,thestimuli,andrelated
information about the experiment. The project will maintain a common format for each kind of
experimental data, and establish a data processing pipeline that, when possible, is shared across
labsinthedi ff erentIRGs.Thepipelinewillyielddatathatisstoredinanopen-source fi lestructure
to facilitate sharing (e.g., HDF5). Documentation for each kind of fi le will include example scripts
50


<!-- Page 3 -->

(in Python or MatLab) that access the data. Johnson will assist with metadata creation consistent
with international standards (e.g., DataCite metadata schema) to promote discovery, sharing, and
reuse of data. DataCite DOIs will be registered by the CU Boulder CRDDS for all data that is
published via the CU Scholar repository in order to provide persistent and citable links to the data
and metadata. Publishing data with DataCite DOIs supports key open science practices, including
the FAIR data principles.
Plans for data storage: The CU Boulder “PetaLibrary” service provides a stable, scalable, and
cost-e ff ective solution for the storage and archiving of research data. The PetaLibrary enables re-
searchers to build, store, share, and merge large and growing data collections, and is available at a
subsidizedcosttoresearchersa ffi liatedwiththeUniversityofColoradoBoulder.Storageisprovided
in two primary classes: “active” storage for data that is accessed frequently or is still undergoing
analysis, and “archive” storage for data that is more static but needs to be stored securely for a
longer-term. Active storage is accessible from all Research Computing compute resources through
the BeeGFS parallel fi le system. Archive storage resides on a cost- and energy-e ffi cient hierarchical
system backed by LTO tape. Both classes of storage can be accessed by any authorized user from
any Internet-connected computer using several common data-transfer protocols. The PetaLibrary
infrastructure is housed in a controlled-environment data center protected by an uninterruptible
power supply (UPS) system. The PetaLibrary is supported by a high-speed local network and ded-
icated large-bandwidth connections to national research networks through an initially NSF-funded
ScienceDMZ. By using the PetaLibrary as a centralized storage solution, the project team will be
able to collaboratively analyze the data of the project, share it with other collaborators outside of
the core team, and publish and archive data more easily via the Globus connection between the
PetaLibrary and the CU Scholar repository. Non-digital data and other physical resources will be
stored and managed by the individual IRGs.
Active access to the data and compute environment: All Odor2Action collaborators will
have direct access to all compute and data resources funded through this project as well as general
available NSF-funded infrastructure via a ffi liate accounts. The project team has funded compute
nodes in the Blanca cluster for exclusive access as well as a node in the hybrid cloud. All the data
on the PetaLibrary can be shared using Globus. The project team can share the data with anyone
without providing accounts on the CU Boulder system. Globus even allows the project team to
create and manage Globus groups for access to the shared data. Globus also provides a mechanism
for public access to large published data sets on the PetaLibrary via landing pages in the CU
Scholar repository. The project team will have Linux command line access to the compute nodes
purchasedfor thisproject. Thesecomputenodeswill beintegratedintoResearchComputingcondo
computing service “Blanca”. The project team has immediate access to its nodes when needed.
Additionally,theycanutilizefreenodesoftheaggregateBlancacluster.EveryResearchComputing
user has access to the NSF funded RMACC Summit supercomputer for parallel computation in
a shared environment. Home, project, and the PetaLibrary storage is accessible on all computing
resources. Research Computing is providing a JupyterHub and OpenOnDemand environment to
support access to the computing and data resources through Jupyter notebooks and an Rstudio
environment, along with MATLAB. CRDDS and Research Computing will provide training in all
aspects of cyberinfrastructure use, and will support development of data and analytics work fl ows.
Plans for archiving and preservation of access in accordance with open science best
practices: Software and simulation code will be shared and maintained through GitHub, Source-
forge, and ModelDB with persistent and citable versions of software and code archived in Zenodo
(using the GitHub to Zenodo work fl ow). Olfactory bulb imaging data will be hosted at Senselab’s
OdorMapDB. All data generated from experiments and simulations will be archived for up to seven
years (or according to local regulations for non-US based teams). In the interest of conducting
research according to open science best practices, all data supporting publications and of poten-
51


<!-- Page 4 -->

tial reuse by the community will be made publicly accessible and archived inde fi nitely in the CU
Scholar repository at the University of Colorado Boulder. CU Scholar is built on the open-source
Samvera software, and it has a robust preservation plan that guarantees access to and preservation
of data for the lifespan of the repository with post-lifespan contingency plans in place as well.
The repository is in the process of obtaining CoreTrustSeal certi fi cation, a leading certi fi cation for
data repository trustworthiness, and is listed in the Re3data registry of research data repositories.
The repository provides DataCite DOIs for all data sets, which allow data to be discovered via
DataCite’s data set search and API, persistently identi fi ed, and cited. CU Scholar provides public
access via Globus data transfer to data archived on the PetaLibrary for data sets that are too large
to access via direct download. Johnson will assist with preparing data for publication and deposit
in CU Scholar and other repositories in accordance with FAIR data principles, and his team will
provide ongoing data curation for the data published in the CU Scholar repository. Data published
in CU Scholar will provide links to associated publications and archived software and code.
3. Policies
Policies for access and sharing in accordance with open science best practices: All
data will be shared upon publication or one year after funding has ended, whichever comes fi rst.
Data associated with publications and of potential value for reuse by the community will be made
publicly available via the CU Scholar repository, which is openly accessible via HTTP. Large data
sets will be publicly discoverable via CU Scholar and openly accessible via Globus data transfer
from the PetaLibrary. Data sets in CU Scholar are discoverable via the repository’s search/browse
capabilities, indexing in major search engines (e.g., Google), and registries that list CU Scholar
and/or its individual data sets (e.g., Re3data registry of research data repositories and DataCite
data set search). A directory of the non-digital data and other physical resources generated by
the project, including protocols for sharing with team members and external users, will be made
available online via the CU Scholar repository.
Policies and provisions for re-use or re-distribution: No restrictions will be placed on the
data for re-use or re-distribution other than attribution of the data to the original source. This
policy for re-use and redistribution will be included in the human- and machine-readable metadata
that accompanies all published data.
4. Use of Vertebrate Animals: Vertebrateanimals(mice, M. musculus ) will beused ineachof
theIRGs.Miceareanidealmammalianmodelsystembecausetheyarehighlyolfactoryanimalsand
have olfactory circuits that are highly homologous to other vertebrates—including humans—and
because genetic-based tools are available and essential to the proposed experiments. Procedures to
beperformedareprovidedinthedescriptionforeachIRG,andinvolveodorantexposurefollowedby
euthanasia (IRG1); imaging in anesthetized or awake, head- fi xed mice (IRGs1 & 3); measurements
using chronically-implanted probes in freely-moving mice (IRGs 2 & 3); and conditioning of odor-
driven navigation behavior in head- fi xed mice (IRG3). The number of animals to be used in each
study will be determined based on estimates of the success rates of the various experiments and on
statistical power analysis using assumptions of e ff ect sizes, measurement variability and number of
data points that can be collected per experiment. This analysis will be done in consultation with
biostatisticians at each institution; a typical goal will be achieving a statistical power of 0.8 for
comparison of two di ff erent populations with an anticipated medium e ff ect size (Cohen’s d = 0.6),
atacon fi dencelevelof p =0.05.Inallcases,appropriateprotocolswillbeusedtominimizeexposure
to discomfort, pain or injury, including general anesthetics (iso fl urane, ketamine/xylazine), local
anesthetics around incision sites (bupivicaine); administration of post-operative analgesics; and
monitoring of body weight and signs of stress in conditioned animals. Animals will be euthanized
with an overdose of pentobarbital (200 mg/kg), consistent with AVMA Guidelines on Euthanasia
of Animals. All procedures will be approved by Institutional Animal Care and Use Committees.
52