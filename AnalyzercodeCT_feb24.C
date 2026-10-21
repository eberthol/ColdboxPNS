#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TAxis.h"

using namespace std;

// ===================== TV DENOISE (UNCHANGED) =====================
void TV1D_denoise(vector<float>& input, vector<float>& output, const int width, const float lambda) {
  if (width > 0) {
    int k = 0, k0 = 0;
    float umin = lambda, umax = -lambda;
    float vmin = input[0] - lambda, vmax = input[0] + lambda;
    int kplus = 0, kminus = 0;
    const float twolambda = 2.0 * lambda;
    const float minlambda = -lambda;
    for (;;) {
      while (k == width - 1) {
        if (umin < 0.0) {
          do output[k0++] = vmin; while (k0 <= kminus);
          umax = (vmin = input[kminus = k = k0]) + (umin = lambda) - vmax;
        } else if (umax > 0.0) {
          do output[k0++] = vmax; while (k0 <= kplus);
          umin = (vmax = input[kplus = k = k0]) + (umax = minlambda) - vmin;
        } else {
          vmin += umin / (k - k0 + 1);
          do output[k0++] = vmin; while (k0 <= k);
          return;
        }
      }
      if ((umin += input[k + 1] - vmin) < minlambda) {
        do output[k0++] = vmin; while (k0 <= kminus);
        vmax = (vmin = input[kplus = kminus = k = k0]) + twolambda;
        umin = lambda; umax = minlambda;
      } else if ((umax += input[k + 1] - vmax) > lambda) {
        do output[k0++] = vmax; while (k0 <= kplus);
        vmin = (vmax = input[kplus = kminus = k = k0]) - twolambda;
        umin = lambda; umax = minlambda;
      } else {
        k++;
        if (umin >= lambda) {
          vmin += (umin - lambda) / ((kminus = k) - k0 + 1);
          umin = lambda;
        }
        if (umax <= minlambda) {
          vmax += (umax + lambda) / ((kplus = k) - k0 + 1);
          umax = minlambda;
        }
      }
    }
  }
}

// ===================== HELPERS =====================
static bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

static int getChannel(const std::string& path) {
  if (contains(path, "channel0")) return 0;
  if (contains(path, "channel7")) return 7;
  return -1;
}

static std::string getCx(const std::string& path) {
  if (contains(path, "C1")) return "C1";
  if (contains(path, "C2")) return "C2";
  if (contains(path, "C3")) return "C3";
  if (contains(path, "C4")) return "C4";
  return "C?";
}

static std::string getAFE(const std::string& path) {
  for (int i = 0; i <= 9; ++i) {
    std::string key = "AFE" + std::to_string(i);
    if (contains(path, key)) return key;
  }
  return "AFE?";
}

static void getSPEWindow(const std::string& Cx, int ch, double &mean, double &err) {
  // Mean ± Error from your table
  if (Cx=="C1" && ch==0) { mean=353; err=50; return; }
  if (Cx=="C1" && ch==7) { mean=280; err=30; return; }

  if (Cx=="C2" && ch==0) { mean=264; err=50; return; }
  if (Cx=="C2" && ch==7) { mean=226; err=40; return; }

  // if (Cx=="C3" && ch==0) { mean=889; err=100; return; }
  //  if (Cx=="C3" && ch==7) { mean=912; err=100; return; }
  if (Cx=="C3" && ch==0) { mean=245; err=50; return; }
  if (Cx=="C3" && ch==7) { mean=240; err=50; return; }

  if (Cx=="C4" && ch==0) { mean=140; err=40; return; }
  if (Cx=="C4" && ch==7) { mean=247; err=40; return; }

  mean=0; err=1e9;
}

// ===================== PROCESS SINGLE FILE =====================
static void ProcessOneFile(const std::string& inPath) {

  // cosmetic ROOT style (safe)
  gStyle->SetOptStat(0);

  std::string Cx  = getCx(inPath);
  int ch          = getChannel(inPath);
  std::string afe = getAFE(inPath);

  double mean=0.0, err=1e9;
  getSPEWindow(Cx, ch, mean, err);

  // C3: minimum 3; others: 5
  double threshold = 5.0;
  if (Cx == "C1") threshold = 20.0;

  std::string tag     = Cx + "_" + afe + "_ch" + std::to_string(ch);
  std::string outRoot = tag + "_SPE.root";
  std::string outPDF  = tag + "_SPEtemplate.pdf";
  std::string outPDF2D = tag + ((Cx=="C3") ? "_timevssignalSPE_min3.pdf" : "_timevssignalSPE_min5.pdf");

  cout << "\n=== Processing: " << inPath << "\n"
       << "    Output ROOT: " << outRoot << "\n"
       << "    Output PDF : " << outPDF  << "\n"
       << "    2D PDF     : " << outPDF2D << "\n"
       << "    SPE cut    : sum in [" << (mean-err) << ", " << (mean+err) << "]\n"
       << "    2D min     : " << threshold << "\n"
       << endl;

  TFile *f = new TFile(outRoot.c_str(), "RECREATE");

  TH2D *timevssignalSPE =
      new TH2D("timevssignalSPE",";time [ticks];signal [ADC]",
               1000,0,1000,17000,-1000,16000);

  std::ifstream fin(inPath.c_str());
  if (!fin.is_open()) {
    std::cerr << "Cannot open: " << inPath << std::endl;
    f->Close();
    delete f;
    return;
  }

  int intstart = 200;

  std::string line;
  std::vector<float> waveform(1000), input(1000), output(1000);

  while (std::getline(fin, line)) {
    std::istringstream ss(line);
    float val; int idx = 0;
    while (ss >> val && idx < 1000) waveform[idx++] = val;
    if (idx != 1000) continue;

    // baseline: exactly like your simplified version (waveform-based)
    double base = 0.0;
    for (int i = intstart; i < intstart + 180; ++i) base += waveform[i];
    base /= 180.0;

    for (int i = 0; i < 1000; ++i) input[i] = waveform[i];

    TV1D_denoise(input, output, 1000, 5);

    // denoised baseline-subtracted
    double sum = 0.0;
    for (int i = intstart + 182; i < intstart + 210; ++i) {
      double den = output[i] - base;
      sum += den;
    }

    // SPE integral window (table driven)
    if (sum > (mean - err) && sum < (mean + err)) {
      for (int i = 0; i < 1000; ++i) {
        timevssignalSPE->Fill(i, waveform[i] - base);
      }
    }
  }

  // ----------------- "modified histogram" (apply threshold) -----------------
  TH2D *clean = (TH2D*)timevssignalSPE->Clone("timevssignalSPE_clean");

  const int nx = clean->GetNbinsX();
  const int ny = clean->GetNbinsY();
  for (int ix = 1; ix <= nx; ++ix) {
    for (int iy = 1; iy <= ny; ++iy) {
      if (clean->GetBinContent(ix, iy) < threshold) clean->SetBinContent(ix, iy, 0.0);
    }
  }

  // ----------------- get mean waveform via ProfileX (internal helper) -----------------
  // We will NOT save the template as TProfile (to avoid "profile/normalization" confusion).
  // Instead we copy values to a TH1D in ns.
  TProfile *prof_ticks = clean->ProfileX("SPEtemplate_profile_ticks");

  // ----------------- build final template as TH1D in ns, y in ADC -----------------
  const int nb = prof_ticks->GetNbinsX();
  const double xmin_ns = 0.0;
  const double xmax_ns = 1000.0 * 16.0;

  TH1D *SPEtemplate_ns = new TH1D("SPEtemplate_ns",
                                 ";time [ns];ADC",
                                 nb, xmin_ns, xmax_ns);

  for (int ib = 1; ib <= nb; ++ib) {
    SPEtemplate_ns->SetBinContent(ib, prof_ticks->GetBinContent(ib)); // mean ADC
    SPEtemplate_ns->SetBinError(ib,   prof_ticks->GetBinError(ib));
  }

  // ----------------- Write ROOT -----------------
  timevssignalSPE->Write();
  clean->Write();
  prof_ticks->Write();      // keep for debugging (optional but useful)
  SPEtemplate_ns->Write();  // THIS is the final template you want

  // ----------------- PDFs -----------------
  TCanvas *c = new TCanvas("c", "c", 1200, 900);

  // Make sure labels are not cut
  c->SetLeftMargin(0.14);
  c->SetRightMargin(0.14);
  c->SetBottomMargin(0.12);
  c->SetTopMargin(0.06);

  // 2D plot with minimum
  timevssignalSPE->SetMinimum(threshold);
  timevssignalSPE->GetYaxis()->SetTitleOffset(1.25);
  timevssignalSPE->GetXaxis()->SetTitleOffset(1.10);
  timevssignalSPE->Draw("colz");
  c->SaveAs(outPDF2D.c_str());

  // Template plot: draw as HIST (requested)
  c->Clear();
  c->SetLeftMargin(0.16);   // slightly more for "ADC"
  c->SetRightMargin(0.05);
  c->SetBottomMargin(0.12);
  c->SetTopMargin(0.06);

  SPEtemplate_ns->GetYaxis()->SetTitleOffset(1.35);
  SPEtemplate_ns->GetXaxis()->SetTitleOffset(1.10);
  SPEtemplate_ns->Draw("HIST");
  c->SaveAs(outPDF.c_str());

  delete c;

  f->Close();
  delete f;
}

// ===================== MAIN DRIVER (8 FILES HARDCODED) =====================
void AnalyzercodeCT_feb24() {

  std::vector<std::string> files = {

    // C1
    //"C1/C1_Apr112024_6p2V_500Hz_HDconnectedBiased/VGAIN0p4/C1_Apr112024_6p2V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1058_OIsOFF_channel7_AFE0.csv",
   // "C1/C1_Apr112024_6p2V_500Hz_HDconnectedBiased/VGAIN0p4/C1_Apr112024_6p2V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1062_OIsOFF_channel0_AFE0.csv",

    // C2
    //"C2/C2_Apr112024_4p8V_500Hz_HDconnectedBiased/VGAIN0p4/C2_Apr112024_4p8V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1056_OIsOFF_channel0_AFE1.csv",
   // "C2/C2_Apr112024_4p8V_500Hz_HDconnectedBiased/VGAIN0p4/C2_Apr112024_4p8V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1062_OIsOFF_channel7_AFE1.csv",

    // C3
    //  "movedconnections/VGAIN0p4/C3_run18_2024425_EP106_VGAIN0p4_offset1087_OIsOFF_channel0_AFE3.csv",
    //  "movedconnections/VGAIN0p4/C3_run18_2024425_EP106_VGAIN0p4_offset1079_OIsOFF_channel7_AFE3.csv",
   // "C3/C3_Apr112024_5p4V_500Hz_HDconnectedBiased/VGAIN0p4/C3_Apr112024_5p4V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1062_OIsOFF_channel0_AFE2.csv",
   // "C3/C3_Apr112024_5p4V_500Hz_HDconnectedBiased/VGAIN0p4/C3_Apr112024_5p4V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1066_OIsOFF_channel7_AFE2.csv",

    // C4
    "C4/C4_5p5V_500Hz_allChannelsON_MovedConnections_16042024_EP106_VGAIN0p4_offset1100_OIsOFF_channel0_AFE4.csv",
    "C4/C4_5p5V_500Hz_allChannelsON_MovedConnections_16042024_EP106_VGAIN0p4_offset1078_OIsOFF_channel7_AFE4.csv",
    //"movedconnections/VGAIN0p4/C4_5p5V_500Hz_allChannelsON_MovedConnections_16042024_EP106_VGAIN0p4_offset1100_OIsOFF_channel0_AFE4.csv",
   // "movedconnections/VGAIN0p4/C4_5p5V_500Hz_allChannelsON_MovedConnections_16042024_EP106_VGAIN0p4_offset1078_OIsOFF_channel7_AFE4.csv"
    // Corrected with available file: Tluanga- Oct 22, 2026
    // C4 - April 11, 2024 (AFE3)
    "C4/C4_Apr112024_5p5V_500Hz_HDconnectedBiased/VGAIN0p4/C4_Apr112024_5p5V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1075_OIsOFF_channel0_AFE3.csv",
    "C4/C4_Apr112024_5p5V_500Hz_HDconnectedBiased/VGAIN0p4/C4_Apr112024_5p5V_500Hz_HDconnectedBiased_EP106_VGAIN0p4_offset1074_OIsOFF_channel7_AFE3.csv",
  };

  for (const auto &p : files) {
    ProcessOneFile(p);
  }
}
