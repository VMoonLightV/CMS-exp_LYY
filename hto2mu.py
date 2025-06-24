"""
Analysis module: make a H -> MuMu Run3 using NanoAOD
"""
import yaml
import shutil
import logging
import os.path
import pandas as pd
from itertools import chain
import numpy as np

import bamboo.plots
from bamboo import treedecorators as td
from bamboo import treefunctions as op
from bamboo.analysismodules import NanoAODModule, NanoAODHistoModule, HistogramsModule, NanoAODSkimmerModule
from bamboo.scalefactors import get_correction

import utils

logger = logging.getLogger(__name__)


leptonSFLib = {
    "electron_ID": "Electron-ID-SF",
    "electron_reco": "Electron-ID-SF",
    #"electron_trigger": #FIXME, 
    "muon_ID": "NUM_MediumID_DEN_TrackerMuons",
    "muon_iso": "NUM_LoosePFIso_DEN_MediumID",
    "muon_trigger": "NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium",
}

#https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
bTagWorkingPoints = {
    "2022": {
        "btagDeepFlavB": {
            "L"  : 0.0583,
            "M"  : 0.3086,
            "T"  : 0.7183,
            "XT" :  0.8111,  
            "XXT":  0.9512   
        },
    },
    "2022EE": {
        "btagDeepFlavB": { #deepJet
            "L"  : 0.042,
            "M"  : 0.108,
            "T"  : 0.305,
        },
    },
    "2023": {
        "btagDeepFlavB": {
            "L"  : 0.0479,
            "M"  : 0.2431,
            "T"  : 0.6553,
            "XT" : 0.7667,  
            "XXT": 0.9459   
        },
    },
    "2023BPix": {
        "btagDeepFlavB": {
            "L"  : 0.048,
            "M"  : 0.2435,
            "T"  : 0.6563,
            "XT" : 0.7671,  
            "XXT": 0.9483   
        },
    }
}

Var_Binning = {
    "ll_mass": [0, 150.0], 
    "ll_pt": [0.0, 500.0], 
    "ll_eta": [0.0, 10.0], 
    "ll_phi": [0.0, 3.1416],

    "jj_mass": [0.0, 1000.0], 
    "jj_pt": [0.0, 1000.0], 
    "jj_eta": [0.0, 5.0], 
    "jj_phi": [0.0, 3.1416],

    "lljj_mass": [0.0, 1000.0], 
    "lljj_pt": [0.0, 1000.0], 
    "lljj_eta": [0.0, 5.0], 
    "lljj_phi": [0.0, 3.1416],

    "leading_lepton_pt": [0.0, 1000.0],  
    "leading_lepton_eta": [0.0, 5.0], 
    "leading_lepton_phi": [0.0, 3.1416],
    "leading_lepton_mass": [-0.1, 0.1],
    "leading_lepton_charge": [-1., 1],

    "subleading_lepton_pt": [0.0, 1000.0],  
    "subleading_lepton_eta": [0.0, 5.0], 
    "subleading_lepton_phi": [0.0, 3.1416],
    "subleading_lepton_mass": [-0.1, 0.1],
    "subleading_lepton_charge": [-1., 1.],

    "dEta_2lep": [0.0, 50.0],  
    "dPhi_2lep": [0.0, 5.0], 
    "dRmm_2lep": [0.0, 10.10],

    "dEta_jj_abs": [0.0, 10.0], 
    "dPhi_jj": [0.0, 5.0],

    "leading_jet_pt": [0.0, 1000.0], 
    "leading_jet_eta": [0.0, 5.0], 
    "leading_jet_phi": [0.0, 3.1416],
    "subleading_jet_pt": [0.0, 1000.0],  
    "subleading_jet_eta": [0.0, 5.0], 
    "subleading_jet_phi": [0.0, 3.1416],
}

def inputs_variables(ll_p4, jj_p4, dilepton, jets, muons, electrons):
    lljj_p4 = ll_p4 + jj_p4
    skim_inputs = {
            "ll_mass": ll_p4.M(),
            "ll_pt"  : ll_p4.Pt(), 
            "ll_eta" : op.abs(ll_p4.Eta()),
            "ll_phi" : op.abs(ll_p4.Phi()),
            
            "jj_mass": jj_p4.M(),
            "jj_pt"  : jj_p4.Pt(),
            "jj_eta" : op.abs(jj_p4.Eta()),
            "jj_phi" : op.abs(jj_p4.Phi()),
            
            "lljj_mass": lljj_p4.M(),   
            "lljj_pt"  : lljj_p4.Pt(),   
            "lljj_eta" : op.abs(lljj_p4.Eta()), 
            "lljj_phi" : op.abs(lljj_p4.Phi()),
            
            #for further analysis
            "leading_lepton_pt"       : dilepton[0].pt,
            "leading_lepton_eta"      : op.abs(dilepton[0].eta),
            "leading_lepton_phi"      : op.abs(dilepton[0].phi),
            "leading_lepton_charge"   : dilepton[0].charge,
            "leading_lepton_mass"     : dilepton[0].mass,
            "subleading_lepton_pt"    : dilepton[1].pt,
            "subleading_lepton_eta"   : op.abs(dilepton[1].eta),
            "subleading_lepton_phi"   : op.abs(dilepton[1].phi),
            "subleading_lepton_charge": dilepton[1].charge,
            "subleading_lepton_mass"  : dilepton[1].mass,
            
            #symmetric variable has an absolute value added
            "dEta_2lep": op.abs(dilepton[0].eta - dilepton[1].eta),
            "dPhi_2lep": op.abs(op.deltaPhi(dilepton[0].p4, dilepton[1].p4)),
            "dRmm_2lep": op.deltaR(dilepton[0].p4, dilepton[1].p4),
            
            #"dEta_jj"    : jets[0].eta-jets[1].eta,#op.abs
            "dEta_jj_abs": op.abs(jets[0].eta-jets[1].eta),#but already abs here?
            "dPhi_jj"    : op.abs(op.deltaPhi(jets[0].p4, jets[1].p4)),
            #"dPhi_jj_mod"    : (op.deltaPhi(jets[0].p4, jets[1].p4) + np.pi) % (2 * np.pi) - np.pi,
            #"dPhi_jj_mod_abs": op.abs((op.deltaPhi(jets[0].p4, jets[1].p4) + np.pi) % (2 * np.pi) - np.pi),
            
            "leading_jet_pt"    : jets[0].pt,
            "leading_jet_phi"   : op.abs(jets[0].phi),
            "leading_jet_eta"   : op.abs(jets[0].eta),
            "subleading_jet_pt" : jets[1].pt,
            "subleading_jet_eta": op.abs(jets[1].eta),
            "subleading_jet_phi": op.abs(jets[1].phi),
            
            # Min/Max delta R between a lepton and jet
            #"dRmin_lepjet" : op.rng_min((muons, jets) lambda mu,j : op.deltaR(mu, j)) 
            #"dRmax_lepjet" : op.rng_max((muons, jets) lambda mu,j : op.deltaR(mu, j)) 
            #"dRmax_2lepjet": op.rng_min((ll_p4, jets) lambda mu,j : op.deltaR(mu, j)) 
            #"dRmax_2lepjet": op.rng_max((ll_p4, jets) lambda mu,j : op.deltaR(mu, j)) 
            }
    return skim_inputs


def pogEraFormat(era):
    if '2023' in era or '2022' in era: return era[:4] +'_Summer'+era.replace('20','')
    else: return era.replace("UL", "") + "_UL" 


# https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
def localizePOGSF(era, POG, fileName):
    subdir = pogEraFormat(era)
    return os.path.join("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration", "POG", POG, subdir, fileName)


def bTagDef(jets, era, wp="M", tagger="btagDeepFlavB"):
    return op.select(jets, lambda jet: getattr(jet, tagger) >= bTagWorkingPoints[era][tagger][wp])


# from https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
def flagDef(flags, era, isMC):
    cuts = [
        flags.goodVertices,
        flags.globalSuperTightHalo2016Filter,
        flags.EcalDeadCellTriggerPrimitiveFilter,
        flags.BadPFMuonDzFilter,
        flags.BadPFMuonFilter,
        flags.eeBadScFilter
    ]
    if '2023' in era or '2022' in era:
        cuts.append(flags.hfNoisyHitsFilter)
    return cuts


def getScaleFactor(era, noSel, systName, wp=None, defineOnFirstUse=True):
    correction = leptonSFLib[systName]
    if "muon" in systName:
        fileName = localizePOGSF(era, "MUO", f"muon_Z.json.gz")
        etaParam = "eta" #"abseta"
        etaExpr = lambda mu: op.abs(mu.eta)
    elif "electron" in systName:
        fileName = localizePOGSF(era, "EGM", "electron.json.gz")
        etaParam = "eta"
        etaExpr = lambda el: el.eta + el.deltaEtaSC
        if era =='2023': year = "2023PromptC"
        elif era =='2023BPix': year = "2023PromptD"
        elif era =='2022': year = "2022Re-recoBCD"
        elif era =='2022EE': year = "2022Re-recoE+PromptFG"
        else: year = era.replace("UL", "")
    
    if "muon" in systName:
        return get_correction(fileName, correction, 
                              params={"pt": lambda mu: mu.pt, etaParam: etaExpr},
                              systParam="scale_factors",
                              systNomName="nominal",
                              systVariations={f"{systName}up": "systup", f"{systName}down": "systdown"},
                              defineOnFirstUse=defineOnFirstUse, sel=noSel)
    if "electron" in systName:
        if '2023' in era: return get_correction(fileName, correction, 
                              params={"pt": lambda el: el.pt, etaParam: etaExpr, "year": year, "WorkingPoint": wp,"phi": lambda el: el.phi},
                              systParam="ValType", systNomName="sf",
                              systVariations={f"{systName}up": "sfup", f"{systName}down": "sfdown"},
                              defineOnFirstUse=defineOnFirstUse, sel=noSel)
        else: return get_correction(fileName, correction, 
                              params={"pt": lambda el: el.pt, etaParam: etaExpr, "year": year, "WorkingPoint": wp},
                              systParam="ValType", systNomName="sf",
                              systVariations={f"{systName}up": "sfup", f"{systName}down": "sfdown"},
                              defineOnFirstUse=defineOnFirstUse, sel=noSel)


def getNanoAODDescription(era, isMC, metName='MET', doRocCor=True):
    nanoJetMETCalc_both = td.CalcCollectionsGroups(
            Jet=("pt", "mass"), changes={metName: (f"{metName}T1", f"{metName}T1Smear")},**{metName: ("pt", "phi")})
    nanoJetMETCalc_data = td.CalcCollectionsGroups(
            Jet=("pt", "mass"), changes={metName: (f"{metName}T1",)}, **{metName: ("pt", "phi")})
    systVars=[td.nanoFatJetCalc ] +[nanoJetMETCalc_both if isMC else nanoJetMETCalc_data]
    if doRocCor:systVars.append(td.nanoRochesterCalc)
    return td.NanoAODDescription.get("v12", year=era[:4], isMC=isMC, systVariations=systVars)


class NanoHMuMuBase(NanoAODModule):
    """ Base module for NanoAOD H->MuMu example """
    
    def addArgs(self, parser):
        super().addArgs(parser)
        parser.add_argument("--test", action="store_true", help="Test max. 1 MC and 1 data file (to test before launching a large job submission)")
        parser.add_argument("--backend", type=str, default="dataframe", help="Backend to use, 'dataframe' (default), 'lazy', or 'compiled'")
        parser.add_argument("--samples", nargs='*', required=False, help="Sample template YML file")
        # add jes jer correction later
        parser.add_argument("--jes-scheme", choices=["All", "Merged", "Total"], default="Total", help="JEC uncertainty scheme (default: %(default)s)")
        parser.add_argument("--split-jer", action="store_true", default=False, help="Produce the split JER variations")
        parser.add_argument("--doMETT1Smear", action="store_true", default=False, help="do T1 MET smearing")
        parser.add_argument("--roc-corr", action="store_true", help="Enable muon Rochester correction")
        parser.add_argument("--skim", action="store_true", default=False, help="Save selected branches for events that pass the selection to a skimmed tree")
        parser.add_argument("--evaluate-mva", action="store_true", default=False, help="Save selected branches for events that pass the selection to a skimmed tree")
        parser.add_argument("--btag-sf", choices=['none', 'fixWP', 'itFit'], default='none', help="Choose b-tag SFs to use (default: %(default)s)")
        parser.add_argument("--decorr-btag", action="store_true", help="Decorrelate b-tagging uncertainties for different working points")
        parser.add_argument("-s", "--systematics", action="store_true", default=False, help="Produce systematic variations")

    '''
    #FIXME --samples still does not work (Note on 11.3)
    def _loadSampleLists(self, analysisCfg):
        # fill sample template using JSON files
        if "samples" not in analysisCfg:
            eras = self.args.eras[1]
            samples = {}
            # make sure we use absolute paths as this argument will be used by the worker jobs
            self.args.samples = [ os.path.abspath(p) for p in self.args.samples ]
            for tmpPath in self.args.samples:
                with open(tmpPath) as f_:
                    template = yaml.load(f_, Loader=yaml.SafeLoader)
                    samples.update(utils.fillSampleTemplate(template, eras))
            analysisCfg["samples"] = samples

    def customizeAnalysisCfg(self, analysisCfg):
        self._loadSampleLists(analysisCfg)
        samples = analysisCfg["samples"]
        print("customizeAnalysisCfg analysisCfg: ", analysisCfg["samples"])
        # reduce job splitting
        #if self.args.reduce_split:
        #    for smp in samples.values():
        #        smp["split"] *= self.args.reduce_split

        # if we're not doing systematics, remove the systematics samples from the list
        #if not (self.doSysts and self.args.syst_samples):
        #    for smp in list(samples.keys()):
        #        if "syst" in samples[smp]:
        #            samples.pop(smp)

        if not self.args.distributed or self.args.distributed != "worker":
            if self.args.test:
                # only keep 1 MC (if possible, a signal) and 1 data file, for testing the plotter
                chosenEra = self.args.eras[1][0] if self.args.eras[1] else None
                foundMC = utils.getAnyMCSample(samples, self.isMC, era=chosenEra, signalSample=True)
                if foundMC is None:
                    logger.info("No signal sample found for testing, falling back on background sample")
                    foundMC = utils.getAnyMCSample(samples, self.isMC, era=chosenEra, signalSample=False)
                if foundMC is None:
                    logger.warning("No MC sample found for testing!")
                else:
                    logger.info(f"Found MC sample for testing: {foundMC}")
                    chosenEra = samples[foundMC]["era"]
                foundData = utils.getAnyDataSample(samples, self.isMC, era=chosenEra)
                if foundData is None:
                    logger.warning("No data sample found for testing!")
                else:
                    logger.info(f"Found data sample for testing: {foundData}")
                    if chosenEra is None:
                        chosenEra = samples[foundData]["era"]
                for smpNm in list(samples.keys()):
                    if smpNm != foundMC and smpNm != foundData:
                        samples.pop(smpNm)
                # only keep 1 file per sample
                self.args.maxFiles = 1
                # adjust the eras in the analysis config
                for era in list(analysisCfg["eras"].keys()):
                    if era != chosenEra:
                        analysisCfg["eras"].pop(era)
                logger.info(f"Testing mode: only using one file; only running on era {chosenEra}; using data: {foundData}; using MC: {foundMC}")

            # back up analysis config - not really needed since git config is stored, but JIC
            newCfg = os.path.join(self.args.output, "full_analysis.yml")
            os.makedirs(self.args.output, exist_ok=True)
            yaml.add_representer(str, utils.yaml_latex_representer)
            with open(newCfg, "w") as f_:
                yaml.dump(analysisCfg, f_, default_flow_style=False, indent=4)
    '''

    def prepareTree(self, tree, sample=None, sampleCfg=None, backend=None):
        from bamboo.analysisutils import makeMultiPrimaryDatasetTriggerSelection
        from bamboo.analysisutils import configureJets, configureType1MET, configureRochesterCorrection
        from bamboo.analysisutils import makePileupWeight
        
        era = sampleCfg.get("era") if sampleCfg else None
        
        isMC = self.isMC(sample)
        isNotWorker = (self.args.distributed != "worker")
        
        splitJER = self.args.split_jer
        jesScheme = self.args.jes_scheme 
        configMET = self.args.doMETT1Smear
        
        self.metName = "PuppiMET"

        # Decorate the tree
        tree, noSel, be, lumiArgs = super().prepareTree(tree, sample=sample, sampleCfg=sampleCfg,
            description=getNanoAODDescription(era, isMC, self.metName, doRocCor=self.args.roc_corr) 
            ,backend=self.args.backend )
        
        # always-on event weights
        if isMC:
            noSel = noSel.refine("mcWeight", weight=tree.genWeight)

            if '2023BPix' in era: goldenJSON = "Collisions2023_369803_370790_eraD_GoldenJson"
            elif '2023' in era: goldenJSON = "Collisions2023_366403_369802_eraBC_GoldenJson"
            elif '2022EE' in era: goldenJSON = "Collisions2022_359022_362760_eraEFG_GoldenJson"
            else: goldenJSON = "Collisions2022_355100_357900_eraBCD_GoldenJson"
            
            puTuple = (localizePOGSF(era, "LUM", "puWeights.json.gz"), goldenJSON)
            self.PUWeight = makePileupWeight(puTuple, tree.Pileup_nTrueInt, systName="pileup", sel=noSel)
            noSel = noSel.refine("puWeight", weight= self.PUWeight)
        
        ## JEC and JER (data and mc) 
        JECTagDatabase = {
                "2022"       : "Summer22_22Sep2023_V2_MC",
                "2022EE"     : "Summer22EE_22Sep2023_V2_MC",
                "Run2022C"   : "Summer22_22Sep2023_RunCD_V2_DATA",
                "Run2022D"   : "Summer22_22Sep2023_RunCD_V2_DATA",
                "Run2022E"   : "Summer22EE_22Sep2023_RunE_V2_DATA",
                "Run2022F"   : "Summer22EE_22Sep2023_RunF_V2_DATA",
                "Run2022G"   : "Summer22EE_22Sep2023_RunG_V2_DATA",
                
                "2023"        : "Summer23Prompt23_V1_MC",
                "2023BPix"    : "Summer23BPixPrompt23_V1_MC",
                "Run2023Cv123": "Summer23Prompt23_RunCv123_V1_DATA",
                "Run2023Cv4"  : "Summer23Prompt23_RunCv4_V1_DATA",
                "Run2023D"    : "Summer23BPixPrompt23_RunD_V1_DATA",
                }

        JERTagDatabase = {
                "2022"       : "Summer22_22Sep2023_JRV1_MC",
                "2022EE"     : "Summer22EE_22Sep2023_JRV1_MC",
                
                "2023"       : "Summer23Prompt23_RunCv1234_JRV1_MC",
                "2023BPix"   : "Summer23BPixPrompt23_RunD_JRV1_MC",
                }
        
        
        if isMC: # only mc for now
            exclJetSysts = []
            if '2023' in era: run3 = ["2023"]
            else: run3 = ["2022"]
            
            if splitJER:
                exclJetSysts += [f"jer{x}" for x in range(2, 6) ]
            
            if jesScheme == "Merged":
                sources = [s for e in run3 for s in ['Regrouped_Absolute', f'Regrouped_Absolute_{e}', 
                            'Regrouped_BBEC1', f'Regrouped_BBEC1_{e}', 
                            'Regrouped_EC2', f'Regrouped_EC2_{e}', 
                            'Regrouped_FlavorQCD', 
                            'Regrouped_HF', f'Regrouped_HF_{e}', 
                            'Regrouped_RelativeBal', f'Regrouped_RelativeSample_{e}'] ] 
                exclJetSysts += [ f"Regrouped_jesEC2_{e}" for e in run3 ]
            
            elif jesScheme == "All":
                # here we specify explicitly the sources we use, to avoid having the jetMet 
                # calculator pre-compute all the sources we won't use use in the end
                sources = ["AbsoluteStat", "AbsoluteScale", "AbsoluteMPFBias",
                           "Fragmentation", "SinglePionECAL", "SinglePionHCAL",
                           "FlavorQCD",
                           "FlavorPureGluon", "FlavorPureQuark", "FlavorPureCharm", "FlavorPureBottom",
                           "TimePtEta", "RelativeJEREC1", "RelativePtBB", "RelativePtEC1",
                           "RelativeBal", "RelativeSample", "RelativeFSR", "RelativeStatFSR", "RelativeStatEC",
                           "PileUpDataMC", "PileUpPtRef", "PileUpPtBB", "PileUpPtEC1"]
            else:
                sources = ["Total"]

            exclJetSysts += list(chain.from_iterable([ (f"{j}up", f"{j}down") for j in exclJetSysts ]))
            
            cmJMEArgs = {
                "jsonFile": localizePOGSF(era, "JME", "jet_jerc.json.gz"),
                "jec": JECTagDatabase[era],
                "smear": JERTagDatabase[era],
                "splitJER": splitJER,
                "isMC": isMC,
                "backend": be, 
                "jecLevels":[], 
                "addHEM2018Issue":(era == "2018UL"),
                "jesUncertaintySources": sources,
                "enableSystematics": (lambda v: v not in exclJetSysts),
                "jsonFileSmearingTool": os.path.join("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration", "POG", "JME", "jer_smear.json.gz"),
                #"regroupTag": "V2",
                #"uName": sampleName,
                #"mayWriteCache":isNotWorker,
                }
            
            configureJets(tree._Jet, jetType="AK4PFPuppi", **cmJMEArgs)
            if configMET:
                configureType1MET(getattr(tree, f"_{self.metName}T1"), **cmJMEArgs)
            
        triggersPerPrimaryDataset = {
                    "Muon"      : [ #tree.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, 
                                    tree.HLT.IsoMu24, 
                                    #tree.HLT.Mu15_IsoVVVL_PFHT450, 
                                    #tree.HLT.CascadeMu100, 
                                    #tree.HLT.HighPtTkMu100, 
                                    ], 
                    "SingleMuon": [ tree.HLT.IsoMu24, 
                                    #tree.HLT.IsoMu27, 
                                    #tree.HLT.Mu50, 
                                    #tree.HLT.HighPtTkMu100, 
                                    #tree.HLT.CascadeMu100,
                                    ],
                    #"DoubleMuon": [ #tree.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL,  # this one is prescaled
                    #                #tree.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ,
                    #                #tree.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8,
                    #                tree.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, 
                    #                ],
                    #"MuonEG"    : [
                    #                tree.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL,
                    #                tree.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ,
                    #                tree.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL,
                    #                tree.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ, 
                    #                tree.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL,
                    #                tree.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ 
                    #                ],
                    }

    
        if self.isMC(sample):
            noSel = noSel.refine("withtriggers", cut=(op.OR(*chain.from_iterable(triggersPerPrimaryDataset.values()))))
        else:
            noSel = noSel.refine("withtriggers", cut=(makeMultiPrimaryDatasetTriggerSelection(sample, triggersPerPrimaryDataset)))

        noSel = noSel.refine("METflags", cut=flagDef(tree.Flag, era, self.isMC(sample)))

        # top pt reweighting
        if self.isMC(sample) and sample.startswith("TTto"):
            def top_pt_weight(pt):
                return op.exp(-2.02274e-01 + 1.09734e-04*pt + -1.30088e-07*pt**2 + (5.83494e+01/(pt+1.96252e+02)))

            def getTopPtWeight(tree):
                lastCopy = op.select(
                    tree.GenPart, lambda p: (op.static_cast("int", p.statusFlags) >> 13) & 1)
                tops = op.select(lastCopy, lambda p: p.pdgId == 6)
                antitops = op.select(lastCopy, lambda p: p.pdgId == -6)
                weight = op.switch(op.AND(op.rng_len(tops) >= 1, op.rng_len(antitops) >= 1),
                                   op.sqrt(top_pt_weight(
                                       tops[0].pt) * top_pt_weight(antitops[0].pt)),
                                   1.)
                return weight

            logger.info(
                "Applying Top Pt reweighting (only for TTbar samples)")

            noSel = noSel.refine("topPt", weight=op.systematic(
                getTopPtWeight(tree)))

        return tree, noSel, be, lumiArgs


class NanoHMuMu(NanoHMuMuBase, HistogramsModule):
    """ Example module: H->MuMu histograms from NanoAOD """
    
    def definePlots(self, t, noSel, sample=None, sampleCfg=None):
    
        from bamboo.plots import CutFlowReport, SummedPlot, Skim, Plot
        from bamboo.plots import EquidistantBinning as EqBin
        from bamboo.analysisutils import forceDefine

        era = sampleCfg.get("era") if sampleCfg else None
        isMC = self.isMC(sample)
        
        binScaling = 1
        plots = []
        
        cfr = CutFlowReport("yields", recursive=True, printInLog=True)
        plots.append(cfr)
        
        cfr.add(noSel, "with trigger")
        
        noSel = noSel.refine("GoodPV", cut=[t.PV.npvsGood > 0])
        cfr.add(noSel, "with good reconstructed primary vertices")

        self.met = t.MET #getattr(t, f"_{self.metName}")

        # muons
        self.sorted_muons = op.sort(t.Muon, lambda mu : -mu.pt)
        self.muons = op.select(self.sorted_muons, lambda mu : op.AND(mu.pt > 20., 
                                                                     op.abs(mu.eta) < 2.4, 
                                                                     mu.mediumId, 
                                                                     mu.pfRelIso04_all<0.25, 
                                                                     op.abs(mu.sip3d) < 4.))
        # electrons
        self.sorted_electrons = op.sort(t.Electron, lambda ele : -ele.pt)
        self.electrons = op.select(self.sorted_electrons, lambda ele : op.AND(ele.pt > 20., op.abs(ele.eta) < 2.5 , ele.cutBased>=3, op.abs(ele.sip3d) < 4., 
                                                                              op.OR(op.AND(op.abs(ele.dxy) < 0.05, op.abs(ele.dz) < 0.1), 
                                                                                    op.AND(op.abs(ele.dxy) < 0.05, op.abs(ele.dz) < 0.2) ))) 
        # jets
        #self.jets_noclean = op.select(t.Jet, lambda j: op.OR(op.AND(j.jetId & 0x2, op.abs(j.eta) < 4.7, j.pt > 26.)))
        self.jets_noclean = op.select(t.Jet, lambda j: op.OR(op.AND(j.jetId & 0x2, op.abs(j.eta) < 2.6, j.pt > 26.),
                                      op.AND(j.jetId & 0x2, op.abs(j.eta) < 3.1,op.abs(j.eta) > 2.6, j.pt > 50.), 
                                      op.AND(j.jetId & 0x2, op.abs(j.eta) < 4.7,op.abs(j.eta) > 3.1, j.pt > 26.)))
        self.jets_sorted = op.sort(self.jets_noclean, lambda j: -j.pt)
            
        # bjets selection
        self.btagger = "btagDeepFlavB"
        self.bJetsM  = bTagDef(self.jets_sorted, era, "M", self.btagger)
        self.bJetsL  = bTagDef(self.jets_sorted, era, "L", self.btagger)
        
        # Di-leptons selection 
        osdilep = lambda lep1,lep2 : op.AND(lep1.charge != lep2.charge, op.in_range(70., op.invariant_mass(lep1.p4, lep2.p4), 150.))

        os_dileptons_comb = {
                "MuMu" : op.combine(self.muons, N=2, pred= osdilep),
                #"MuEl" : op.combine((self.muons, self.electrons), pred=lambda mu,ele : op.AND(ele.charge != mu.charge)),
                }

        one_os_dilepton = lambda cmbRng : op.AND(op.rng_len(cmbRng) > 0, cmbRng[0][0].pt > 26.)
        has_os_ll = noSel.refine("hasOSLL", cut=op.OR(*( one_os_dilepton(rng) for rng in os_dileptons_comb.values())))
        
        if self.isMC(sample):
            for calcProd in t._Jet.calcProds:
                forceDefine(calcProd, has_os_ll)
        
        if self.isMC(sample): 
            def muonTriggerSF(mu):
                sf = getScaleFactor(era, noSel, systName="muon_trigger", defineOnFirstUse=False)
                return op.switch( mu.pt >= 26. , sf(mu), op.c_float(1.)) 
            
            def scalefactor(wp):
                return getScaleFactor(era, noSel, systName="electron_reco", wp=wp, defineOnFirstUse=False)
            
            def eleRecoSF(el):
                return op.multiSwitch( (el.pt <20. , scalefactor('RecoBelow20')(el)),
                                       (op.in_range(20, el.pt, 75), scalefactor('Reco20to75')(el)), 
                                       scalefactor('RecoAbove75')(el))
            
            muonIDSF = getScaleFactor(era, noSel, systName="muon_ID", defineOnFirstUse=True)
            muonIsoSF = getScaleFactor(era, noSel, systName="muon_iso", defineOnFirstUse=True)
            eleIDSF = getScaleFactor(era, noSel, systName="electron_ID", wp="Tight", defineOnFirstUse=False)
        
            leptons_scalefactor = { "MuMu" : (lambda ll : [ muonTriggerSF(ll[0]), #muonTriggerSF(ll[1]),
                                                            muonIDSF(ll[0]), muonIDSF(ll[1]),
                                                            muonIsoSF(ll[0]), muonIsoSF(ll[1])
                                                            ] ),
                                    #"MuEl" : (lambda ll : [ muonTriggerSF(ll[0]), #eleTriggerSF(ll[1]),
                                                            #muonIDSF(ll[0]), eleIDSF(ll[1]),
                                                            #muonIsoSF(ll[0]), eleRecoSF(ll[1])
                                                            #] ),
                                } 

        # Create categories based on dilepton channels
        categories = {
            channel: (
                cat_ll_rng[0],
                has_os_ll.refine(
                    f"hasOs{channel}",
                    cut=one_os_dilepton(cat_ll_rng),
                    weight=(leptons_scalefactor[channel](cat_ll_rng[0]) if self.isMC(sample) else None) 
                )
            )
            for channel, cat_ll_rng in os_dileptons_comb.items()
        }
        
        self.jets= op.select(self.jets_sorted,lambda j: op.AND(
                                op.NOT(op.rng_any(self.muons, lambda l: op.deltaR(l.p4, j.p4) < 0.4)),
                                #op.NOT(op.rng_any(self.electrons, lambda l: op.deltaR(l.p4, j.p4) < 0.4))
                                #not clean jet for electron now
                            ))
        
        for channel, (dilepton, dilepSel) in categories.items():
          
            cfr.add(dilepSel, "2 OS lep.(%s) + $m_{ll}$ cut in $channel")
            
            ll_p4   = dilepton[0].p4  + dilepton[1].p4
            jj_p4   = self.jets[0].p4 + self.jets[1].p4
            lljj_p4 = ll_p4 + jj_p4
            
            for i in range(2):
                plots += [ Plot.make1D(f"{channel}_lep{i+1}_{nm}", var, dilepSel, binning,
                    title=f"{utils.getCounter(i+1)} lepton {title}", plotopts=utils.getOpts(channel))
                for nm, (var, binning, title) in {
                    "pt" : (dilepton[i].pt,  EqBin(60 // binScaling, 0., 500.), "P_{T} (GeV)"),
                    "eta": (dilepton[i].eta, EqBin(50 // binScaling, -2.5, 2.5), "#eta"),
                    "phi": (dilepton[i].phi, EqBin(50 // binScaling, -3.1416, 3.1416), "#phi")
                    }.items()
                ]
                plots += [ Plot.make1D(f"{channel}_jet{i+1}_{nm}", var, dilepSel, binning,
                    title=f"{utils.getCounter(i+1)} jet {title}", plotopts=utils.getOpts(channel))
                for nm, (var, binning, title) in {
                    "pt" : (self.jets[i].pt,  EqBin(60 // binScaling, 0., 1000.), "P_{T} (GeV)"),
                    "eta": (self.jets[i].eta, EqBin(50 // binScaling, -5., 5.0), "#eta"),
                    "phi": (self.jets[i].phi, EqBin(50 // binScaling, -3.1416, 3.1416), "#phi")
                    }.items()
                ]
            
            # MET plots 
            for i in range(2):
                plots.append(Plot.make1D(f"{channel}_MET_lep{i+1}_deltaPhi",
                       op.Phi_mpi_pi(dilepton[i].phi - self.met.phi), dilepSel, EqBin(60 // binScaling, -3.1416, 3.1416),
                       title="#Delta #phi (lepton, MET)", plotopts=utils.getOpts(channel, **{"log-y": False})))
                MT = op.sqrt( 2. * self.met.pt * dilepton[i].p4.Pt() * (1. - op.cos(op.Phi_mpi_pi(self.met.phi - dilepton[i].p4.Phi()))) )
                plots.append(Plot.make1D(f"{channel}_lep{i+1}_MET_MT", MT, dilepSel,
                        EqBin(60 // binScaling, 0., 600.), title="Lepton M_{T} (GeV)",
                        plotopts=utils.getOpts(channel)))
            
            plots.append(Plot.make1D(f"{channel}_MET_pt", self.met.pt, dilepSel,
                    EqBin(60 // binScaling, 0., 600.), title="MET p_{T} (GeV)",
                    plotopts=utils.getOpts(channel)))
            plots.append(Plot.make1D(f"{channel}_MET_phi", self.met.phi, dilepSel,
                    EqBin(60 // binScaling, -3.1416, 3.1416), title="MET #phi",
                    plotopts=utils.getOpts(channel, **{"log-y": False})))
                      
            # primary and secondary vertices             
            plots += [ Plot.make1D(f"{channel}_number_primary_reconstructed_vertices",
                            t.PV.npvsGood, dilepSel,
                            EqBin(50 // binScaling, 0., 100.), title="reconstructed vertices",
                            plotopts=utils.getOpts(channel, **{"log-y": True}))]
            
            plots += [ Plot.make1D(f"{channel}_secondary_vertices_{nm}",
                            op.map(t.SV, vVar), dilepSel, binning,
                    title=f"SV {nm}", plotopts=utils.getOpts(channel, **{"log-y": True}))
                    for nm, (vVar, binning, title) in {
                        "mass" : (lambda sv : sv.mass, EqBin(50 // binScaling, 0., 150.), "m_{sv} (GeV)"),
                        "eta"  : (lambda sv : sv.eta , EqBin(50 // binScaling,-4.0, 4.0), "#eta"),
                        "phi"  : (lambda sv : sv.phi , EqBin(50 // binScaling, -3.1416, 3.1416), "#phi"),
                        "pt"   : (lambda sv : sv.pt  , EqBin(50 // binScaling, 0., 1000.), "P_{T} (GeV)")
                        }.items()
                    ] 
            
            ###############
            # SR selections 
            ###############
            if channel != 'MuEl' :
                # missing b-tag SF, comment it out for the moment
                # ttH_Sel = dilepSel.refine(f"ttH_Sel_{channel}", cut=op.OR(op.rng_len(self.bJetsM) >0, op.rng_len(self.bJetsL) >1)) 
                # not quiet sure yet about VH
                # And not use VH yet
                #VH_Sel  = dilepSel.refine(f"VH_Sel_{channel}", cut=op.OR(op.rng_len(self.muons) >=3, op.rng_len(self.electrons) >= 1 )) 
                VBF_Sel = dilepSel.refine(f"VBF_Sel_{channel}", cut=op.AND(  
                                            op.rng_len(self.jets) >= 2,   
                                            self.jets[0].pt > 35.,  self.jets[1].pt > 26., 
                                            op.abs(self.jets[0].eta - self.jets[1].eta) > 2.5,  
                                            op.invariant_mass(self.jets[0].p4, self.jets[1].p4) > 400.  ))

                for  cat, sel in { 'All' : dilepSel,
                                   #'ttH' : ttH_Sel,
                                   #'VH'  : VH_Sel.refine(f"VH_ExclusiveSel_{channel}", cut=op.NOT(ttH_Sel.cut)), 
                                   #'VBF' : VBF_Sel.refine(f"VBF_ExclusiveSel_{channel}", cut=op.NOT(ttH_Sel.cut)),
                                   'VBF' : VBF_Sel,
                                   #'ggH' : dilepSel.refine(f"ggH_ExclusiveSel_{channel}", cut=[ op.NOT(ttH_Sel.cut), op.NOT(VBF_Sel.cut)]),
                                   #'ggH' : dilepSel.refine(f"ggH_ExclusiveSel_{channel}", cut=op.NOT(VBF_Sel.cut))
                                }.items():
                                
                    cfr.add(sel, f"{cat}_{channel}_sel")
                    
                    plots += [ Plot.make1D(f"{channel}_{cat}_dilepton_{nm}", var, sel, binning,
                              title=f"dilepton {title}", plotopts=utils.getOpts(channel))
                               for nm, (var, binning, title) in {
                                  "pt" : (ll_p4.Pt() , EqBin(50 // binScaling, 0., 1000.), "P_{T} (GeV)"),
                                  "eta": (ll_p4.Eta(), EqBin(50 // binScaling, -10., 10.), "#eta"),
                                  "phi": (ll_p4.Phi(), EqBin(50 // binScaling, -3.1416, 3.1416), "#phi"),
                                  "mass": (ll_p4.M(), EqBin(50 // binScaling, 70., 150.), "m_{ll} (GeV)"),
                                  "njet": (op.rng_len(self.jets), EqBin(11 // binScaling, -0.5, 10.5), "njets"),
                                  }.items()
                              ]
  
                    for reg, massWindow in { #"ZCR": [70., 100.],
                                             #"LSideband": [100., 115.],
                                             #"SR": [115., 135.],
                                             #"RSideband": [135., 150.],
                                             "Z":[70,110],
                                             "SR":[110,150],
                                             "LSR":[100,150],
                                            }.items():

                        finalSel =  sel.refine(f"{channel}_{cat}_{reg}", cut=op.in_range(massWindow[0], ll_p4.M(), massWindow[1]))
                        cfr.add(finalSel, f"{channel}_{cat}_{reg}_sel")
                        
                        plots += [ Plot.make1D(f"{channel}_mass_{cat}_{reg}", ll_p4.M(), finalSel, EqBin(50 // binScaling, massWindow[0], massWindow[1]),
                                            title=f"{reg}", plotopts=utils.getOpts(channel))
                                    ]
                    ###############
                    # SKIM
                    ###############
                        if cat == 'VBF':# only for test, or it will report "cling JIT session error: Cannot allocate memory"
                            if self.args.skim and reg=="LSR":
                              skim_inputs = inputs_variables(ll_p4, jj_p4, dilepton, self.jets, self.muons, self.electrons)
                              skim_inputs.update({
                                          # just copy the variable as it is in the nanoAOD input
                                          "run"            : None,
                                          "event"          : None,
                                          "luminosityBlock": None,
                                          # needed inputs 
                                          'era'            : op.c_int(int(era[:4])),
                                          'total_weight'   : finalSel.weight,
                                          'PU_weight'      : self.PUWeight if isMC else op.c_float(1.), 
                                          'MC_weight'      : t.genWeight if isMC else op.c_float(1.),
                                          })
      
                              plots.append(Skim(f"{channel}_{cat}_{reg}_sel", skim_inputs, finalSel)) 
  
                    ###############
                    # DNN 
                    ###############
                            if self.args.skim:
                              hhmodel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DNN/', 'Keras_BYS_dense_model.onnx')) 
                              mvaEvaluator = op.mvaEvaluator(hhmodel_path, mvaType='ONNXRuntime',otherArgs="output", nameHint='onnx_hhmodel')
                              skim_inputs  = inputs_variables(ll_p4, jj_p4, dilepton, self.jets, self.muons, self.electrons)
                              
                              DNN_Inputs   = op.array('float',*[op.static_cast('float',val) for val in skim_inputs.values()])
                              DNN_Output   = mvaEvaluator(DNN_Inputs)

                              plotOptions={ "blinded-range" : [1.0, 1.0],}
                              plotOptions_bkg={ "blinded-range" : [1.0, 1.0],}
                              if reg=="SR":
                                plotOptions={ "blinded-range" : [0.5, 1.0],}
                                plotOptions_bkg={ "blinded-range" : [0., 0.5],}
                                            
                              plots += [Plot.make1D(f'DNNOutput_{channel}_signal_{cat}_{reg}_node', DNN_Output[0], finalSel, 
                                          EqBin(20 // binScaling, 0., 1.), title=f'signal DNN {cat} {reg} output node', plotopts=plotOptions)]              
                                            
                              '''
                              plots += [Plot.make1D(f'DNNOutput_{channel}_SoB_{cat}_{reg}_node', DNN_Output[0]/DNN_Output[1], finalSel, 
                                          EqBin(20 // binScaling, 0., 1.), title=f'signal over background DNN {cat} {reg} output node', plotopts=plotOptions)]
                              plots += [Plot.make1D(f'DNNOutput_{channel}_signal_{cat}_{reg}_node', DNN_Output[0], finalSel, 
                                          EqBin(20 // binScaling, 0., 1.), title=f'signal DNN {cat} {reg} output node', plotopts=plotOptions)]
                              plots += [Plot.make1D(f'DNNOutput_{channel}_background_{cat}_{reg}_node', DNN_Output[1], finalSel, 
                                          EqBin(20 // binScaling, 0., 1.), title=f'background DNN {cat} {reg} output node', plotopts=plotOptions_bkg)]
                              plots += [ Plot.make1D(f"DNN{VBFDNNvar}_{channel}_{cat}_{reg}_node", varfunc, finalSel, EqBin(20 // binScaling, Var_Binning[VBFDNNvar][0], Var_Binning[VBFDNNvar][1]), title=f'DNN var {VBFDNNvar} {cat} {reg}', plotopts=utils.getOpts(channel))
                               for  VBFDNNvar, varfunc in skim_inputs.items()
                              ]
                              '''
                              

                           
        return plots


    def postProcess(self, taskList, config=None, workdir=None, resultsdir=None):
        super(NanoHMuMu, self).postProcess(taskList, config, workdir, resultsdir )
        
        from bamboo.analysisutils import loadPlotIt
        from bamboo.plots import Skim, Plot, DerivedPlot
        from bamboo.root import gbl
 
        self.plotList = self.getPlotList(resultsdir=resultsdir, config=config)

        plotList_2D = [ap for ap in self.plotList if ( isinstance(ap, Plot) or isinstance(ap, DerivedPlot) ) and len(ap.binnings) == 2 ]
        logger.debug("Found {0:d} plots to save".format(len(plotList_2D)))

        p_config, samples, plots_2D, systematics, legend = loadPlotIt(config, plotList_2D, 
                    eras=self.args.eras[1], workdir=workdir, resultsdir=resultsdir, readCounters=self.readCounters, 
                    vetoFileAttributes=self.__class__.CustomSampleAttributes, plotDefaults=self.plotDefaults)

        #outDir = os.path.join(resultsdir, "normalizedSummedSignal")
        #if os.path.isdir(outDir): 
        #    shutil.rmtree(outDir)
        #os.makedirs(outDir)
        
        #if self.normalizeForCombine:
        if True:
            plotstoNormalized = []
            for plots in self.plotList:
                plotstoNormalized.append(plots)
            if not os.path.isdir(os.path.join(resultsdir, "normalizedForCombined")):
                os.makedirs(os.path.join(resultsdir,"normalizedForCombined"))
    
            if plotstoNormalized:
                utils.normalizeAndMergeSamplesForCombined(plotstoNormalized, self.readCounters, config, resultsdir, os.path.join(resultsdir, "normalizedForCombined"))

        if self.args.skim:
            logger.info("Required to save skim")
            skims = [ap for ap in self.plotList if isinstance(ap, Skim)]
            print("skims:", skims)
            if skims:
                try:
                    for skim in skims:
                        frames = []
                        for smp in samples:
                            for cb in (smp.files if hasattr(smp, "files") else [smp]):  # could be a helper in plotit
                                # Take specific columns
                                tree = cb.tFile.Get(skim.treeName)
                                if not tree:
                                    print( f"KEY TTree {skim.treeName} does not exist, we are gonna skip this {smp}\n")
                                else:
                                    N = tree.GetEntries()
                                    cols = gbl.ROOT.RDataFrame(cb.tFile.Get(skim.treeName)).AsNumpy()
                                    cols["total_weight"] *= cb.scale
                                    cols["process"] = [smp.name]*len(cols["total_weight"])
                                    frames.append(pd.DataFrame(cols))
                        df = pd.concat(frames)
                        df["process"] = pd.Categorical(df["process"], categories=pd.unique(df["process"]), ordered=False)
                        pqoutname = os.path.join(resultsdir, f"{skim.name}.parquet")
                        df.to_parquet(pqoutname)
                        logger.info(f"Dataframe for skim {skim.name} saved to {pqoutname}")
                except ImportError as ex:
                    logger.error("Could not import pandas, no dataframes will be saved")


class SkimNanoHMuMu(NanoHMuMuBase, NanoAODSkimmerModule):
    """ Base module for NanoAOD H->MuMu Skimmer """
    def defineSkimSelection(self, tree, noSel, sample=None, sampleCfg=None):
        
        muons = op.select(tree.Muon, lambda mu: op.AND(mu.pt > 20., op.abs(mu.eta) < 2.4))
        hasTwoMu = noSel.refine("hasTwoMu", cut=(op.rng_len(muons) >= 2))
        
        varsToKeep = {"nMuon": None, "Muon_eta": None, "Muon_pt": None}  # from input file
        varsToKeep["nSelMuons"] = op.static_cast("UInt_t", op.rng_len(muons))  # TBranch doesn't accept size_t
        varsToKeep["selMuons_i"] = muons.idxs
        varsToKeep["selMu_miniPFRelIsoNeu"] = op.map(
            muons, lambda mu: mu.miniPFRelIso_all - mu.miniPFRelIso_chg)
        
        return hasTwoMu, varsToKeep

