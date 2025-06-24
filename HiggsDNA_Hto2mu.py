from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.selections.lepton_selections_Hmm import (
    select_muons_Hmm,
    select_electrons_Hmm,
#    select_photons_zmmy,
    get_Hmm,
)
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    dump_ak_array,
    dress_branches,
)
from higgs_dna.selections.jet_selections import (select_jets,find_VBF,get_jj,select_genjets,gen_match,bjets_cut)


# from higgs_dna.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.tools.flow_corrections import calculate_flow_corrections

from typing import Any, Dict, List, Optional
import awkward as ak
import numpy as np
import warnings
import vector
import logging
import functools
import operator
from collections import defaultdict
from coffea.analysis_tools import Weights
from coffea.analysis_tools import PackedSelection
import hist

logger = logging.getLogger(__name__)
vector.register_awkward()


class HmmProcessor(HggBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleMuon.*",
        analysis="HmmAnalysis",
        skipCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        cross_section:Dict[str, float] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet",
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            output_location=output_location,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            skipCQR=skipCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            cross_section = cross_section,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format,
        )
        
#        print("\n\n\nccccccccccccccccccccc\n\n\n")
#        print(year)
#        print(cross_section)
#        print("\n\n\nccccccccccccccccccccc\n\n\n")
        
        self.trigger_group = ".*DoubleMuon.*"       #change trigger
        self.analysis = "HmmAnalysis"
        # muon selection cuts
        self.muon_pt_threshold = 20.0
        self.muon_max_eta = 2.4
        self.muon_wp = "mediumId"
        self.muon_max_pfRelIso04_all = 0.2
        self.global_muon = False
        
        self.electron_pt_threshold = 20.0
        self.electron_max_eta = 2.4
        self.electron_wp = "mediumId"
        self.electron_max_pfRelIso04_all = 0.2
        self.global_electron = False
#        self.min_farmuon_pt = 20
        self.min_dimuon_mass = 70.0
        self.max_dimuon_mass = 200.0
        # photon selection cuts
#        self.photon_pt_threshold = 20.0

        # mumugamma selection cuts
        self.Hmass = 125.0
#        self.min_mmy_mass = 60
#        self.max_mmy_mass = 120
#        self.max_mm_mmy_mass = 180
#        self.max_fsr_photon_dR = 0.8
        
        self.jet_pt_threshold = 26.0
        self.jet_max_eta = 4.7
        self.jet_ele_min_dr = 0.4
        self.jet_muo_min_dr = 0.4
        
        self.jet_deta_threshold = 2.5
        self.min_dijet_mass = 400.0
        self.max_dijet_mass = 100000.0



    def apply_metfilters(self, events: ak.Array) -> ak.Array:
        # met filters
        met_filters = self.meta["flashggMetFilters"][self.data_kind]
        filtered = functools.reduce(
            operator.and_,
            (events.Flag[metfilter.replace("Flag_", "")] for metfilter in met_filters),
        )

        return filtered

    def apply_triggers(self, events: ak.Array) -> ak.Array:
        trigger_names = []
        print("\n\n")
        print("trigger:")
        triggers = self.meta["TriggerPaths"][self.trigger_group][self.analysis]
        print(triggers)
        print("\n\n")
        hlt = events.HLT
        for trigger in triggers:
            actual_trigger = trigger.replace("HLT_", "").replace("*", "")
            for field in hlt.fields:
                if field == actual_trigger:
                    trigger_names.append(field)
        triggered = functools.reduce(
            operator.or_, (hlt[trigger_name] for trigger_name in trigger_names)
        )

        return triggered

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset = events.metadata["dataset"]
        
#        print("\n\n\nddddddddddddddddddddd\n\n\n")
#        print(events)
#        print(self.year[dataset])
#        print(self.cross_section[dataset])
#        print("\n\n\nddddddddddddddddddddd\n\n\n")
        
        eve_sel = PackedSelection()

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        run_summary = defaultdict(float)
        run_summary[dataset] = {}
        if self.data_kind == "mc":
            run_summary[dataset]["nTot"] = int(ak.num(events.genWeight, axis=0))
            run_summary[dataset]["nPos"] = int(ak.sum(events.genWeight > 0))
            run_summary[dataset]["nNeg"] = int(ak.sum(events.genWeight < 0))
            run_summary[dataset]["nEff"] = int(
                run_summary[dataset]["nPos"] - run_summary[dataset]["nNeg"]
            )
            run_summary[dataset]["genWeightSum"] = float(ak.sum(events.genWeight))
        else:
            run_summary[dataset]["nTot"] = int(len(events))
            run_summary[dataset]["nPos"] = int(run_summary[dataset]["nTot"])
            run_summary[dataset]["nNeg"] = int(0)
            run_summary[dataset]["nEff"] = int(run_summary[dataset]["nTot"])
            run_summary[dataset]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            lumimask = select_lumis(self.year[dataset][0], events, logger)
            events = events[lumimask]
            # try:
            #     lumimask = select_lumis(self.year[dataset][0], events, logger)
            #     events = events[lumimask]
            # except:
            #     logger.info(
            #         f"[ lumimask ] Skip now! Unable to find year info of dataset: {dataset}"
            #     )

        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(ak.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        # events = events[self.apply_metfilters(events)]
        
        print("Is there Smear_sigma_m?",self.Smear_sigma_m)
        print("Is there apply trigger?",self.apply_trigger)
        if self.apply_trigger:
            trig_flag = self.apply_triggers(events)
            events = events[trig_flag]
        print("\nevent count after trigger", len(events),"\n\n\n\n")

            # events = events[self.apply_triggers(events)]

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
#        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset]
        except KeyError:
            systematic_names = []

        # object corrections:
        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                varying_function = available_object_corrections[correction_name]
                events = varying_function(events=events, year=self.year[dataset][0])
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        # select muons
        muons = events.Muon
        good_muons = muons[select_muons_Hmm(self, muons)]
        dimuons = ak.combinations(good_muons, 2, fields=["lead", "sublead"])
        sel_dimuons = (
            (dimuons["lead"].charge * dimuons["sublead"].charge == -1)
            & (dimuons["lead"].pt > 20.0)
        )
        good_dimuons = dimuons[sel_dimuons]
        n_good_dimuon = ak.sum(sel_dimuons, axis=1)
        
        #!!!just for jet horn study
        eve_sel.add("n_dimuon", n_good_dimuon > 0)



        events["mm"] = get_Hmm(self, good_dimuons)
             
        sel_mm = ak.fill_none(ak.ones_like(events["mm"].dimuon.lead.pt) > 0, False)
        eve_sel.add("mm_finder", sel_mm)
        
        n_mm_events = events[eve_sel.all("n_dimuon")] 
        print("\nevent count after n_dimuon>0:", len(n_mm_events),"\n\n\n\n")
        
        good_mm_events = events[eve_sel.all("mm_finder")]
        print("\nevent count after mm_finder:", len(good_mm_events),"\n\n\n\n")
        #print("skip goodmm\n\n\n\n")
        
        '''
        print("genJetIdxG columns:", events.Jet.genJetIdxG[0])
        print("hfadjacentEtaStripsSize columns:", events.Jet.hfadjacentEtaStripsSize[0])
        print("hfsigmaPhiPhi columns:", events.Jet.hfsigmaPhiPhi[0])
        print("area columns:", events.Jet.area[0])
        '''
        
        
        #select electrons
        electrons = events.Electron
        good_electrons = electrons[select_electrons_Hmm(self, electrons)]
        
        
        
        events["nleps"] = ak.num(good_muons, axis=1) + ak.num(good_electrons, axis=1)
        
        
        #select jets
        print("######################################################")
        # ´òÓ¡ËùÓÐ¶¥²ãÁÐÃû
        print("type:",self.data_kind)
        print("era: ",self.year[dataset][0])
        #print("pt columns:", events.Jet.pt[0])

        
        #print("genJetIdx columns:", events.Jet.genJetIdx[0])
        Jets = events.Jet
        
        base_fields = {
                            "pt": Jets.pt,
                            "eta": Jets.eta,
                            "phi": Jets.phi,
                            "mass": Jets.mass,
                            "charge": ak.zeros_like(
                                Jets.pt
                            ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                            "hFlav": Jets.hadronFlavour
                            if self.data_kind == "mc"
                            else ak.zeros_like(Jets.pt),
                            "btagDeepFlav_B": Jets.btagDeepFlavB,
                            "btagDeepFlav_CvB": Jets.btagDeepFlavCvB,
                            "btagDeepFlav_CvL": Jets.btagDeepFlavCvL,
                            "btagDeepFlav_QG": Jets.btagDeepFlavQG,
                            "btagPNetB": Jets.btagPNetB,
                            "btagPNetQvG": Jets.btagPNetQvG,
                            "PNetRegPtRawCorr": Jets.PNetRegPtRawCorr,
                            "PNetRegPtRawCorrNeutrino": Jets.PNetRegPtRawCorrNeutrino,
                            "PNetRegPtRawRes": Jets.PNetRegPtRawRes,
                            
                            "jetId": Jets.jetId,
                            
                            "chHEF": Jets.chHEF,
                            "neHEF": Jets.neHEF,
                            "chEmEF": Jets.chEmEF,
                            "neEmEF": Jets.neEmEF,
                            "nConstituents": Jets.nConstituents,
                            
                            
                            "nMuons":Jets.nMuons,
                            "nSVs":Jets.nSVs,
                            "muonIdx1":Jets.muonIdx1,
                            "muonIdx2":Jets.muonIdx2,
                            "svIdx1":Jets.svIdx1,
                            "svIdx2":Jets.svIdx2,
                            
                            
        }
        '''
        "btagRobustParTAK4B": Jets.btagRobustParTAK4B,
        "btagRobustParTAK4CvB": Jets.btagRobustParTAK4CvB,
        "btagRobustParTAK4CvL": Jets.btagRobustParTAK4CvL,
        "btagRobustParTAK4QG": Jets.btagRobustParTAK4QG,
        '''
        
        if self.data_kind == "mc" :
            base_fields.update({
                "genJetIdx": Jets.genJetIdx,
                "genJetIdxG": Jets.genJetIdxG,
                "hfadjacentEtaStripsSize": Jets.hfadjacentEtaStripsSize,
                "hfsigmaPhiPhi": Jets.hfsigmaPhiPhi,
                "area": Jets.area,

            })
            
        if self.year[dataset][0] != '2024':
            base_fields.update({
                "btagRobustParTAK4B": Jets.btagRobustParTAK4B,
            })
        
        jets = ak.zip(base_fields)
        
        
                
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        
        #ttH selection is added here
        print("processing btag jet cut\n\n\n")
        btag_L_jets_noclean=jets[
            bjets_cut(self, jets, "L", self.year[dataset][0] )
        ]
        
        btag_M_jets_noclean=jets[
            bjets_cut(self, jets, "M", self.year[dataset][0] )
        ]
        
        btag_T_jets_noclean=jets[
            bjets_cut(self, jets, "T", self.year[dataset][0] )
        ]
        
        print("finish btag jet cut\n\n\n")
        
        # two btag jet score : max
        btag_sorted_jets = jets[ak.argsort(jets.btagDeepFlav_B, axis=1, ascending=False)]

        padded_jets = ak.pad_none(btag_sorted_jets, 2)  
        top1 = padded_jets.btagDeepFlav_B[:, 0]
        top2 = padded_jets.btagDeepFlav_B[:, 1]
        sum_top2_all = ak.fill_none(top1, 0) + ak.fill_none(top2, 0)
        
        print("finish btag jet score\n\n\n")
        
        
        
        
        
        # for VBF below ##########################
        
        jets = jets[
            select_jets(self, jets, good_muons, good_electrons)
        ]
        jets = jets[ak.argsort(jets.pt, ascending=False)]
        
        dijets = ak.combinations(jets, 2, fields=["lead", "sublead"])
        
        
        
        
        events["VBF_jj"] = find_VBF(self, dijets)
        events["nVBFdijets"] = ak.where(ak.is_none(events["VBF_jj"]), 0, 1)
        events["njets"] = ak.num(jets, axis=1)
        
        events["btag_Medium_jets"] = ak.num(btag_M_jets_noclean, axis=1)
        events["btag_Loose_jets"] = ak.num(btag_L_jets_noclean, axis=1)
        events["btag_Tight_jets"] = ak.num(btag_T_jets_noclean, axis=1)
        events["btag_score_2jets_top"] = sum_top2_all
        print("sum_topscore length",len(sum_top2_all))
        
        # for dijet, as the entries of dijet is different from dimuon
        # in order to generate root file, set -1 for the None value
        # we don't care about the -999 ones
        events["jj"] = get_jj(self, dijets)
        
        
        if self.data_kind == "mc" :    

            GenPart=events.GenPart
          
            genpart= ak.zip(
                  {
                  "pt":GenPart.pt,
                  "eta": GenPart.eta,
                  "phi": GenPart.phi,
                  "mass": GenPart.mass,
                  "pdgId": GenPart.pdgId,
                  "status": GenPart.status
                  }
                  )
            genpart = ak.with_name(genpart, "PtEtaPhiMCandidate")
                  
            #print("Raw status flags:", events.GenPart.statusFlags, "\n\n")
            genpart_status = (
                    # events.GenPart.status == 62   # what should status be? 23? 33? 62?
                    events.GenPart.hasFlags(['isPrompt', 'isLastCopy'])# not sure what it is used for
                )

            gentop_mask = (
                    (genpart.pdgId == 6) | (genpart.pdgId == -6)
                ) & genpart_status
            
            print("ak.num(genpart[gentop_mask]):",ak.num(genpart[gentop_mask]),"\n\n")
            if True:#ak.sum(ak.num(genpart[gentop_mask])) != 0:
                    gentops = ak.pad_none(genpart[gentop_mask], 2)  
                    print("top 1pt:",gentops[:, 0].pt,"\n\n")
                    print("top 1:",gentops[:, 0],"\n\n")
                    fake_top= ak.zip(
                                        {
                                            "pt": -999.0,
                                            "phi": -999.0,
                                            "eta": -999.0,
                                            "mass":-999.0,
                                            "pdgId":6
                                        },
                                        with_name="fakeone",
                                    ) 
                    print("top 1:",gentops[:, 0],"\n\n")
                    gentops = ak.fill_none(gentops, fake_top)   
                    print("top 1pt:",gentops[:, 0].pt,"\n\n")
                        
                    gentops = gentops[ak.argsort(gentops.pt, axis=1, ascending=False)]
                    print("top 1pt:",gentops[:, 0].pt,"\n\n")
                    
                    gentops["charge"] = ak.where(
                        gentops.pdgId == 6, 2 / 3, 0
                    ) + ak.where(gentops.pdgId == -6, -2 / 3, 0)

                    
                    
                      
                    properties = ["pt", "eta", "phi", "mass"]
                    default_value = -999.0
                    
                    events["gentop1_pt_test"] = ak.fill_none(gentops.pt[:, 0], default_value)
                    print("gentop1 ", gentops.pt[:, 0],"\n\n")
                    for i in range(2): 
                        top = gentops[:, i]
                        for prop in properties:
                            key = f"gentop{i+1}_{prop}"
                            events[key] = ak.fill_none(top[prop], default_value)
                            print("filled ",key)
                    print("nice filling!###################\n\n\n")
    
                  
        # for fat jet
        FatJet=events.FatJet
          
        FatJet= ak.zip(
                  {
                  "pt":FatJet.pt,
                  "eta": FatJet.eta,
                  "phi": FatJet.phi,
                  "mass": FatJet.mass,
                  "particleNetWithMass_TvsQCD": FatJet.particleNetWithMass_TvsQCD,
                  "particleNetWithMass_HbbvsQCD": FatJet.particleNetWithMass_HbbvsQCD
                  }
                  )
        fatjets = ak.with_name(FatJet, "PtEtaPhiMCandidate")
        
        print("true fatjets num:",ak.sum(ak.num(fatjets)),"\n\n")
        fatjets = ak.pad_none(fatjets, 2) 
        fake_fatjet= ak.zip({
                          "pt": -999.0,
                          "phi": -999.0,
                          "eta": -999.0,
                          "mass":-999.0,
                          "particleNetWithMass_TvsQCD":-999.0,
                          "particleNetWithMass_HbbvsQCD":-999.0
                          },
                          with_name="fakeone1",
                        ) 
        fatjets = ak.fill_none(fatjets, fake_fatjet) 
        fatjets = fatjets[ak.argsort(fatjets.pt, ascending=False)]
        
        
        for i in range(2): 
            for prop in ["pt", "eta", "phi", "mass","particleNetWithMass_TvsQCD","particleNetWithMass_HbbvsQCD"]:
                key = f"fatjet{i+1}_{prop}"
                #print(f"fatjet{i+1}_{prop}:",fatjets[:,i][prop] )
                events[key] = fatjets[:,i][prop]

          

        
        events["dijet_pt"]=ak.fill_none(events["jj"].obj_dijet.pt, -999)
        events["dijet_eta"]=ak.fill_none(events["jj"].obj_dijet.eta, -999)
        events["dijet_phi"]=ak.fill_none(events["jj"].obj_dijet.phi, -999)
        events["dijet_mass"]=ak.fill_none(events["jj"].obj_dijet.mass, -999)
        
        #!!!just for jet horn study
        sel_jet = ak.fill_none(ak.ones_like(events["jj"].dijet.lead.pt) > 0, False)
        eve_sel.add("jj_finder", sel_jet)
        
        # ttH selection
        sel_ttH = sel_ttH = ak.fill_none(
                    (events.btag_Medium_jets > 0) | (events.btag_Loose_jets > 1), 
                    False
                )
        #eve_sel.add("ttH_finder", sel_ttH)
        
        

        
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/ustcfs3/CMSUser/ylou/HiggsDNA/scripts/test/event_statistics.txt"
        
   
        with open(output_file, "a") as f:
            
            f.write(f"time:{timestamp}\n")
            print(f"dataset:{dataset}")
            print("era: ",self.year[dataset][0])
            f.write(f"[dataset] : {dataset}\n")
            f.write(f"[Trigger] : {len(events)}\n")
            print("\nevent count after trigger", len(events),"\n\n")
        
            good_mm_events = events[eve_sel.all("mm_finder")]
            f.write(f"[mm_finder] : {len(good_mm_events)}\n")
            print("\nevent count after mm_finder:", len(good_mm_events),"\n\n")
            
            jj_events = events[eve_sel.all("jj_finder")]
            f.write(f"[jj_finder] : {len(jj_events)}\n")
            print("\nevent count after jj_finder", len(jj_events),"\n\n")
        
            events_cond2_3 = events[eve_sel.all("mm_finder", "jj_finder")]
            f.write(f"[mm_finder & jj_finder]: {len(events_cond2_3)}\n")
            print("\nevent count after good_mm and jj_finder", len(events_cond2_3),"\n\n\n\n")
            
            rate1 =len(events_cond2_3)/len(events)
        
            f.write("=== summary===\n")
            f.write(f"final rate: {rate1:.2%}\n\n")

        # !!! and have bug here
        #events["dijet_deta"]=ak.fill_none(events["jj"].obj_dijet.deta, -999)
        #events["dijet_dphi_abs"]=ak.fill_none(events["jj"].obj_dijet.dphi_abs, -999)
        
        # as we only use lljj_vars for VBF DNN analysis, we don't us them for now
        
        
        
        
        # after all cuts
        events = events[eve_sel.all(*(eve_sel.names))]
        if len(events) == 0:
            logger.debug("No surviving events in this run, return now!")
            return run_summary

        print("\n\nEntries:")
        print(ak.num(events.mm.dimuon.lead.pt, axis = 0))
        print("\n\n")

        # fill ntuple
        if self.output_location is not None:
            ntuple = {}
            ntuple["event"] = events.event
            ntuple["cross_section"] = np.full(len(events), self.cross_section[dataset])
            
            ntuple["lumi"] = events.luminosityBlock
            ntuple["run"] = events.run
            ntuple = dress_branches(ntuple, events.PV, "PV")
            ntuple = dress_branches(ntuple, events.Rho, "Rho")
            if self.data_kind == "mc":
                ntuple = dress_branches(ntuple, events.Pileup, "Pileup")
                
            # muon info 
            for order in ["lead", "sublead"]:
                for var in ["pt", "eta", "phi", "mass", 
                            "tunepRelPt", "charge", "dxy",
                            "dz", "svIdx"]:
                            
                    name=f"muon_{order}_{var}"
                    ntuple[name]=events.mm.dimuon[order][var]

            ntuple["muon_lead_tunep_pt"] = (
                events.mm.dimuon.lead.pt * events.mm.dimuon.lead.tunepRelPt
            )

            ntuple["muon_sublead_tunep_pt"] = (
                events.mm.dimuon.sublead.pt * events.mm.dimuon.sublead.tunepRelPt
            )

            
            ntuple["nleps"] = events.nleps
            ntuple["nVBFdijets"] = events.nVBFdijets
            ntuple["njets"] = events.njets
            
            ntuple["btag_Loose_jets"] = events.btag_Loose_jets
            ntuple["btag_Medium_jets"] = events.btag_Medium_jets
            ntuple["btag_Tight_jets"] = events.btag_Tight_jets
            ntuple["btag_score_2jets_top"] = events.btag_score_2jets_top
            
            

            # dimuon info !!! not able to be usede yet
            for var in ["pt", "eta", "phi", "mass"]:           
                name=f"dimuon_{var}"
                #ntuple[name]=events.mm.obj_dimuon[var]
            
            ntuple["dimuon_pt"] = events.mm.obj_dimuon.pt
            ntuple["dimuon_eta"] = events.mm.obj_dimuon.eta
            ntuple["dimuon_phi"] = events.mm.obj_dimuon.phi
            ntuple["dimuon_mass"] = events.mm.obj_dimuon.mass
            
            
            
            default_value = -999.0
            for order in ["lead", "sublead"]:
                for var in ["pt", "eta", "phi", "mass", 
                            "chHEF", "neHEF", "chEmEF", "neEmEF",
                            "nMuons", "nSVs", "muonIdx1", "muonIdx2",
                            "svIdx1", "svIdx2",
                            "btagDeepFlav_B", "btagDeepFlav_CvB", 
                            "btagDeepFlav_CvL", "btagDeepFlav_QG"]:                    
                    name=f"jet_{order}_{var}"
                    ntuple[name]=ak.fill_none(events["jj"].dijet[order][var], default_value) 
                if self.data_kind == "mc":
                    for var in ["genJetIdx",
                            "genJetIdxG", "hfadjacentEtaStripsSize", 
                            "hfsigmaPhiPhi", "area"]:       
                                 
                      name=f"jet_{order}_{var}"
                      ntuple[name]=ak.fill_none(events["jj"].dijet[order][var], default_value)  
                      #print(f"jet_{order}_{var} length:", len(events["jj"].dijet[order][var]),"\n")
                      
            # genpart top
            if self.data_kind == "mc":
                print("gentop1 ", gentops.pt[:, 0],"\n\n")
                print("gentop1 ", events.gentop1_pt_test,"\n\n")
                print("gentop1 length:", len(events.gentop1_pt_test),"\n\n")
                
                for i in range(2):
                    for prop in ["pt", "eta", "phi", "mass"]: 
                        name = f"GenPart_top{i+1}_{prop}"
                        key = f"gentop{i+1}_{prop}"
                        #print("gentop1 ", getattr(events, key),"\n\n")
                        ntuple[name] = getattr(events, key) 
                
                #!!!!!!!!!!!! problem here too
                #print("should be gentop1", events.gentop1_pt,"\n\n")
                                  

            
            # dijet info !!! not able to be used yet
            for var in ["pt", "eta", "phi", "mass"]:                    
                name=f"dijet_{var}"
                #ntuple[name]=ak.fill_none(events["jj"].obj_dijet[var], -999)
            
            
            ntuple["dijet_pt"] = ak.fill_none(events["jj"].obj_dijet.pt, -999)
            ntuple["dijet_eta"] = ak.fill_none(events["jj"].obj_dijet.eta, -999)
            ntuple["dijet_phi"] = ak.fill_none(events["jj"].obj_dijet.phi, -999)
            ntuple["dijet_mass"] = ak.fill_none(events["jj"].obj_dijet.mass, -999)
            
            #ntuple["dijet_deta"] = events.dijet_deta
            #ntuple["dijet_dphi"] = events.dijet_dphi_abs
            
            for i in range(2): 
                for prop in ["pt", "eta", "phi", "mass","particleNetWithMass_TvsQCD","particleNetWithMass_HbbvsQCD"]:
                    key = f"fatjet{i+1}_{prop}"
                    ntuple[key] = getattr(events, key) 
            
            if self.year[dataset][0] != '2024':
              ntuple["MET_phi"] = events.MET.phi
              ntuple["MET_pt"] = events.MET.pt
              ntuple["MET_significance"] = events.MET.significance
              ntuple["MET_sumEt"] = events.MET.sumEt
  
              ntuple["PuppiMET_phi"] = events.PuppiMET.phi
              ntuple["PuppiMET_pt"] = events.PuppiMET.pt
              ntuple["PuppiMET_sumEt"] = events.PuppiMET.sumEt
  
              ntuple["ChsMET_phi"] = events.ChsMET.phi
              ntuple["ChsMET_pt"] = events.ChsMET.pt
              ntuple["ChsMET_sumEt"] = events.ChsMET.sumEt
              #ntuple["PuppiMET"] = events.PuppiMET
            

        if self.data_kind == "mc":
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            ntuple["dZ"] = events.GenVtx.z - events.PV.z
            ntuple["genWeight"] = events.genWeight
            ntuple["genWeight_sign"] = np.sign(events.genWeight)
            
#            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
#            print(events.genWeight)
#            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")            
            
            
            
        # Fill zeros for data because there is no GenVtx for data, obviously
        else:
            ntuple["dZ"] = ak.zeros_like(events.PV.z)

        # return if there is no surviving events
        if len(ntuple) == 0:
            logger.debug("No surviving events in this run, return now!")
            return run_summary

        if self.data_kind == "mc":
            # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
            event_weights = Weights(size=len(events))
            # _weight will correspond to the product of genWeight and the scale factors
            event_weights._weight = events["genWeight"]

            # corrections to event weights:
            for correction_name in correction_names:
                if correction_name in available_weight_corrections:
                    logger.info(
                        f"Adding correction {correction_name} to weight collection of dataset {dataset}"
                    )
                    varying_function = available_weight_corrections[correction_name]
                    event_weights = varying_function(
                        events=events,
                        weights=event_weights,
                        dataset=dataset,
                        logger=logger,
                        year=self.year[dataset][0],
                    )
            ntuple["weight"] = event_weights.weight()
            ntuple["weight_central"] = event_weights.weight() / events["genWeight"]

            # systematic variations of event weights go to nominal output dataframe:
            for systematic_name in systematic_names:
                if systematic_name in available_weight_systematics:
                    logger.info(
                        f"Adding systematic {systematic_name} to weight collection of dataset {dataset}"
                    )
                    if systematic_name == "LHEScale":
                        if hasattr(events, "LHEScaleWeight"):
                            ntuple["nLHEScaleWeight"] = ak.num(
                                events.LHEScaleWeight,
                                axis=1,
                            )
                            ntuple["LHEScaleWeight"] = events.LHEScaleWeight
                        else:
                            logger.info(
                                f"No {systematic_name} Weights in dataset {dataset}"
                            )
                    elif systematic_name == "LHEPdf":
                        if hasattr(events, "LHEPdfWeight"):
                            # two AlphaS weights are removed
                            ntuple["nLHEPdfWeight"] = (
                                ak.num(
                                    events.LHEPdfWeight,
                                    axis=1,
                                )
                                - 2
                            )
                            ntuple["LHEPdfWeight"] = events.LHEPdfWeight[:, :-2]
                        else:
                            logger.info(
                                f"No {systematic_name} Weights in dataset {dataset}"
                            )
                    else:
                        varying_function = available_weight_systematics[systematic_name]
                        event_weights = varying_function(
                            events=events,
                            weights=event_weights,
                            logger=logger,
                            dataset=dataset,
                            year=self.year[dataset][0],
                        )

                # Store variations with respect to central weight
                if len(event_weights.variations):
                    logger.info(
                        "Adding systematic weight variations to nominal output file."
                    )
                    for modifier in event_weights.variations:
                        ntuple["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )
        # Add weight variables (=1) for data for consistent datasets
        else:
            ntuple["weight_central"] = ak.ones_like(ntuple["event"])
        # to Awkward array: this is necessary, or the saved parquet file is not correct
        ak_ntuple = ak.Array(ntuple)
        
        
        
#        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
#        print(ak_ntuple)
#        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        
        
        
        
        
        if self.output_location is not None:
            fname = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".parquet"
            )
            subdirs = []
            if "dataset" in events.metadata:
                subdirs.append(events.metadata["dataset"])
            dump_ak_array(self, ak_ntuple, fname, self.output_location, metadata, subdirs)
            
#            print("1234")
            
            

        return run_summary

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass




