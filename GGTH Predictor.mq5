//+------------------------------------------------------------------+
//|                                      PredictonFinal2.mq5         |
//|                                              Jason Rusk          |
//|                       For Multi-Timeframe Predictor v7.11        |
//+------------------------------------------------------------------+
#property copyright "Jason Rusk"
#property link      "jason.w.rusk@gmail.com"
#property version   "8.0"
#property description "ML-Based Trading EA with Accuracy Tracking"
#property description "Features: Predicted Price TP, Trailing Stop, Trading Hours, Trend Filter, RSI Filter"

#include <Trade\Trade.mqh>

//--- Input Parameters
input group "=== Testing Mode ==="
input bool InpStrategyTesterMode = true;              // Strategy Tester Mode (use CSV lookups)

input group "=== ML Prediction Settings ==="
input string InpSymbol = "GBPUSD";                     // Symbol to track
input ENUM_TIMEFRAMES InpTradingTimeframe = PERIOD_H1; // Which prediction to use for trading
input bool InpEnableTrading = true;                   // Enable live trading
input double InpRiskPercent = 1.0;                     // Risk per trade (%)
input int InpMinPredictionPips = 20;                   // Min prediction distance to confirm trade (0 = disabled)

input group "=== Take Profit & Stop Loss ==="
input bool InpUsePredictedPrice = false;               // Use predicted price as TP
input int InpStopLossPips = 50;                        // Stop loss in pips
input int InpTakeProfitPips = 100;                     // Take profit in pips (if not using predicted)
input double InpTPMultiplier = 1.0;                    // TP multiplier (adjust predicted TP)
input int InpMinTPPips = 20;                           // Minimum TP distance in pips
input int InpMaxTPPips = 500;                          // Maximum TP distance in pips

input group "=== Trend Filter ==="
input bool InpUseTrendFilter = true;                   // Use trend filter
input int InpTrendMAPeriod = 200;                      // Trend MA period
input ENUM_MA_METHOD InpTrendMAMethod = MODE_EMA;      // Trend MA method
input ENUM_APPLIED_PRICE InpTrendMAPrice = PRICE_CLOSE; // Trend MA price

input group "=== RSI Filter ==="
input bool InpUseRSIFilter = true;                     // Use RSI filter
input int InpRSIPeriod = 14;                           // RSI period
input double InpRSIOverbought = 70.0;                  // RSI overbought level
input double InpRSIOversold = 30.0;                    // RSI oversold level

input group "=== Trailing Stop ==="
input bool InpUseTrailingStop = true;                 // Enable trailing stop
input int InpTrailingStopPips = 12;                    // Trailing stop distance in pips
input int InpTrailingStepPips = 5;                     // Minimum price movement to trail (pips)

input group "=== Trading Days ==="
input bool InpTradeMonday = true;                      // Trade on Monday
input bool InpTradeTuesday = true;                     // Trade on Tuesday
input bool InpTradeWednesday = true;                   // Trade on Wednesday
input bool InpTradeThursday = true;                    // Trade on Thursday
input bool InpTradeFriday = true;                      // Trade on Friday
input bool InpTradeSaturday = false;                   // Trade on Saturday
input bool InpTradeSunday = false;                     // Trade on Sunday

input group "=== Trading Sessions ==="
input bool InpUseSession1 = true;                      // Enable Session 1
input int InpSession1StartHour = 0;                    // Session 1 Start Hour (0-23)
input int InpSession1StartMinute = 0;                  // Session 1 Start Minute (0-59)
input int InpSession1EndHour = 8;                      // Session 1 End Hour (0-23)
input int InpSession1EndMinute = 0;                    // Session 1 End Minute (0-59)

input bool InpUseSession2 = true;                      // Enable Session 2
input int InpSession2StartHour = 8;                    // Session 2 Start Hour (0-23)
input int InpSession2StartMinute = 0;                  // Session 2 Start Minute (0-59)
input int InpSession2EndHour = 16;                     // Session 2 End Hour (0-23)
input int InpSession2EndMinute = 0;                    // Session 2 End Minute (0-59)

input bool InpUseSession3 = true;                      // Enable Session 3
input int InpSession3StartHour = 16;                   // Session 3 Start Hour (0-23)
input int InpSession3StartMinute = 0;                  // Session 3 Start Minute (0-59)
input int InpSession3EndHour = 23;                     // Session 3 End Hour (0-23)
input int InpSession3EndMinute = 59;                   // Session 3 End Minute (0-59)

input group "=== Display Settings ==="
input int InpFontSize = 10;                            // Font size
input color InpTextColor = clrWhite;                   // Text color
input color InpUpColor = clrLimeGreen;                 // Up prediction color
input color InpDownColor = clrRed;                     // Down prediction color
input int InpXOffset = 10;                             // X offset from left
input int InpYOffset = 80;                             // Y offset from top
input bool InpShowDebug = true;                        // Show debug info

//--- Global Variables
CTrade g_trade;
string g_predictionsFile;
string g_statusFile;

// CSV lookup structure
struct CSVPrediction
{
   datetime timestamp;
   double   prediction;
   double   change_pct;
   double   ensemble_std;
};

// CSV data storage
CSVPrediction g_csv_1H[];
CSVPrediction g_csv_4H[];
CSVPrediction g_csv_1D[];
int g_csv_1H_count = 0;
int g_csv_4H_count = 0;
int g_csv_1D_count = 0;

// Prediction data
struct PredictionData
{
   double   prediction;
   double   change_pct;
   double   ensemble_std;
   datetime last_update;
};

PredictionData g_pred_1H;
PredictionData g_pred_4H;
PredictionData g_pred_1D;
double g_current_price = 0;

// Accuracy tracking structure
struct PredictionRecord
{
   datetime timestamp;
   double   predicted_price;
   double   start_price;
   bool     checked;
   bool     accurate;
   datetime check_time;
   string   timeframe_name;
};

// Accuracy tracker
struct AccuracyTracker
{
   int              total_predictions;
   int              accurate_predictions;
   double           accuracy_percent;
   PredictionRecord current_prediction;
};

AccuracyTracker g_tracker_1H;
AccuracyTracker g_tracker_4H;
AccuracyTracker g_tracker_1D;

// Indicator handles
int g_handle_trend_ma = INVALID_HANDLE;
int g_handle_rsi      = INVALID_HANDLE;

// Trade management
datetime g_last_trade_time   = 0;
int      g_min_trade_interval = 3600; // 1 hour minimum between trades
string   g_ea_magic_prefix    = "MLEA_";

datetime g_last_file_check = 0;
datetime g_last_bar_time   = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize files
   g_predictionsFile = "predictions_" + InpSymbol + ".json";
   g_statusFile      = "lstm_status_" + InpSymbol + ".json";

   // Initialize accuracy trackers
   InitAccuracyTracker(g_tracker_1H, "1H");
   InitAccuracyTracker(g_tracker_4H, "4H");
   InitAccuracyTracker(g_tracker_1D, "1D");

   // Load historical accuracy data
   LoadAccuracyData();

   // Initialize indicators
   if(InpUseTrendFilter)
   {
      g_handle_trend_ma = iMA(InpSymbol, Period(), InpTrendMAPeriod, 0, InpTrendMAMethod, InpTrendMAPrice);
      if(g_handle_trend_ma == INVALID_HANDLE)
      {
         Print("ERROR: Failed to create Trend MA indicator");
         return(INIT_FAILED);
      }
   }

   if(InpUseRSIFilter)
   {
      g_handle_rsi = iRSI(InpSymbol, Period(), InpRSIPeriod, PRICE_CLOSE);
      if(g_handle_rsi == INVALID_HANDLE)
      {
         Print("ERROR: Failed to create RSI indicator");
         return(INIT_FAILED);
      }
   }

   // Setup trade object
   g_trade.SetExpertMagicNumber(StringToInteger(g_ea_magic_prefix + "7"));
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   Print("=====================================");
   Print("ML Predictor EA v7.11 Initialized");
   Print("Mode: ", (InpStrategyTesterMode ? "STRATEGY TESTER" : "LIVE"));
   Print("Symbol: ", InpSymbol);
   Print("Trading Timeframe: ", EnumToString(InpTradingTimeframe));
   Print("Live Trading: ", (InpEnableTrading ? "ENABLED" : "DISABLED"));
   Print("-------------------------------------");

   // Load CSV data if in strategy tester mode
   if(InpStrategyTesterMode)
   {
      Print("=====================================");
      Print("STRATEGY TESTER MODE - Loading CSV files");
      Print("=====================================");
      Print("For Strategy Tester, files must be in:");
      Print("  Terminal\\Common\\Files\\");
      Print("");
      Print("Expected files:");
      Print("  - ", InpSymbol, "_1H_lookup.csv");
      Print("  - ", InpSymbol, "_4H_lookup.csv");
      Print("  - ", InpSymbol, "_1D_lookup.csv");
      Print("-------------------------------------");

      bool all_loaded = true;

      if(!LoadCSVLookupFile(PERIOD_H1))
      {
         Print("✗ ERROR: Failed to load 1H CSV file");
         all_loaded = false;
      }
      else
      {
         Print("✓ 1H loaded: ", g_csv_1H_count, " records");
      }

      if(!LoadCSVLookupFile(PERIOD_H4))
      {
         Print("✗ ERROR: Failed to load 4H CSV file");
         all_loaded = false;
      }
      else
      {
         Print("✓ 4H loaded: ", g_csv_4H_count, " records");
      }

      if(!LoadCSVLookupFile(PERIOD_D1))
      {
         Print("✗ ERROR: Failed to load 1D CSV file");
         all_loaded = false;
      }
      else
      {
         Print("✓ 1D loaded: ", g_csv_1D_count, " records");
      }

      if(!all_loaded)
      {
         Print("=====================================");
         Print("CSV LOADING FAILED!");
         Print("=====================================");
         Print("For STRATEGY TESTER mode:");
         Print("  Files MUST be in: Terminal\\Common\\Files\\");
         Print("");
         Print("Full path example:");
         Print("  C:\\Users\\USERNAME\\AppData\\Roaming\\MetaQuotes\\");
         Print("  Terminal\\Common\\Files\\");
         Print("");
         Print("Required files:");
         Print("  ", InpSymbol, "_1H_lookup.csv");
         Print("  ", InpSymbol, "_4H_lookup.csv");
         Print("  ", InpSymbol, "_1D_lookup.csv");
         Print("");
         Print("File format:");
         Print("  - CSV with comma separators");
         Print("  - Header: timestamp,prediction");
         Print("  - Or: timestamp,prediction,change_pct,ensemble_std");
         Print("=====================================");
         return(INIT_FAILED);
      }

      Print("=====================================");
      Print("✓ All CSV files loaded successfully!");
      Print("  Total records: ", (g_csv_1H_count + g_csv_4H_count + g_csv_1D_count));
      Print("=====================================");
   }

   Print("-------------------------------------");
   Print("TP Mode: ", (InpUsePredictedPrice ? "PREDICTED PRICE" : "FIXED PIPS"));
   if(InpUsePredictedPrice)
   {
      Print("  TP Multiplier: ", InpTPMultiplier);
      Print("  Min TP: ", InpMinTPPips, " pips");
      Print("  Max TP: ", InpMaxTPPips, " pips");
   }
   else
   {
      Print("  Fixed TP: ", InpTakeProfitPips, " pips");
   }
   Print("Stop Loss: ", InpStopLossPips, " pips");
   Print("-------------------------------------");
   Print("Trend Filter: ", (InpUseTrendFilter ? "ENABLED" : "DISABLED"));
   if(InpUseTrendFilter)
   {
      Print("  MA Period: ", InpTrendMAPeriod);
      Print("  MA Method: ", EnumToString(InpTrendMAMethod));
   }
   Print("RSI Filter: ", (InpUseRSIFilter ? "ENABLED" : "DISABLED"));
   if(InpUseRSIFilter)
   {
      Print("  RSI Period: ", InpRSIPeriod);
      Print("  Overbought: ", InpRSIOverbought);
      Print("  Oversold: ", InpRSIOversold);
   }
   Print("-------------------------------------");
   Print("Trailing Stop: ", (InpUseTrailingStop ? "ENABLED" : "DISABLED"));
   if(InpUseTrailingStop)
   {
      Print("  Trailing Distance: ", InpTrailingStopPips, " pips");
      Print("  Trailing Step: ", InpTrailingStepPips, " pips");
   }
   Print("-------------------------------------");
   string tradingDays = "";
   if(InpTradeMonday) tradingDays += "Mon ";
   if(InpTradeTuesday) tradingDays += "Tue ";
   if(InpTradeWednesday) tradingDays += "Wed ";
   if(InpTradeThursday) tradingDays += "Thu ";
   if(InpTradeFriday) tradingDays += "Fri ";
   if(InpTradeSaturday) tradingDays += "Sat ";
   if(InpTradeSunday) tradingDays += "Sun";
   Print("Trading Days: ", tradingDays);
   Print("-------------------------------------");
   if(InpUseSession1)
      Print("Session 1: ", IntegerToString(InpSession1StartHour, 2, '0'), ":", IntegerToString(InpSession1StartMinute, 2, '0'),
            " - ", IntegerToString(InpSession1EndHour, 2, '0'), ":", IntegerToString(InpSession1EndMinute, 2, '0'));
   if(InpUseSession2)
      Print("Session 2: ", IntegerToString(InpSession2StartHour, 2, '0'), ":", IntegerToString(InpSession2StartMinute, 2, '0'),
            " - ", IntegerToString(InpSession2EndHour, 2, '0'), ":", IntegerToString(InpSession2EndMinute, 2, '0'));
   if(InpUseSession3)
      Print("Session 3: ", IntegerToString(InpSession3StartHour, 2, '0'), ":", IntegerToString(InpSession3StartMinute, 2, '0'),
            " - ", IntegerToString(InpSession3EndHour, 2, '0'), ":", IntegerToString(InpSession3EndMinute, 2, '0'));

   Print("=====================================");

   // Try to read predictions immediately (for live mode)
   if(!InpStrategyTesterMode)
   {
      if(ReadPredictions())
      {
         Print("✓ Successfully read predictions on startup!");
      }
      else
      {
         Print("⚠ Waiting for prediction files...");
      }
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveAccuracyData();

   // Release indicator handles
   if(g_handle_trend_ma != INVALID_HANDLE)
      IndicatorRelease(g_handle_trend_ma);
   if(g_handle_rsi != INVALID_HANDLE)
      IndicatorRelease(g_handle_rsi);

   // Delete all display objects
   ObjectsDeleteAll(0, "MLEA_");
   ChartRedraw();

   Print("ML Trading EA v7.11 stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Get current bar time
   datetime current_bar_time = iTime(InpSymbol, InpTradingTimeframe, 0);

   // Strategy tester mode - check on new bar
   if(InpStrategyTesterMode)
   {
      if(current_bar_time != g_last_bar_time)
      {
         g_last_bar_time = current_bar_time;

         // Lookup predictions from CSV for current bar
         if(LookupPredictionsFromCSV(current_bar_time))
         {
            // Check accuracy for all timeframes
            CheckAccuracy(g_tracker_1H, PERIOD_H1);
            CheckAccuracy(g_tracker_4H, PERIOD_H4);
            CheckAccuracy(g_tracker_1D, PERIOD_D1);

            // Update display
            UpdateDisplay();

            // Check for trading signals
            if(InpEnableTrading)
            {
               CheckForTradingSignal();
            }
         }
      }
   }
   // Live mode - check file every 5 seconds
   else
   {
      if(TimeCurrent() - g_last_file_check >= 5)
      {
         g_last_file_check = TimeCurrent();

         if(ReadPredictions())
         {
            // Check accuracy for all timeframes
            CheckAccuracy(g_tracker_1H, PERIOD_H1);
            CheckAccuracy(g_tracker_4H, PERIOD_H4);
            CheckAccuracy(g_tracker_1D, PERIOD_D1);

            // Update display
            UpdateDisplay();

            // Check for trading signals
            if(InpEnableTrading)
            {
               CheckForTradingSignal();
            }
         }
         else
         {
            DisplayError();
         }
      }
   }
   
   // Manage trailing stops on every tick
   if(InpEnableTrading)
   {
      ManageTrailingStop();

   }
}

//+------------------------------------------------------------------+
//| Load CSV lookup file                                             |
//+------------------------------------------------------------------+
bool LoadCSVLookupFile(ENUM_TIMEFRAMES timeframe)
{
   string        tf_str = "";
   CSVPrediction temp_array[];
   int           count = 0;

   switch(timeframe)
   {
      case PERIOD_H1: tf_str = "1H"; break;
      case PERIOD_H4: tf_str = "4H"; break;
      case PERIOD_D1: tf_str = "1D"; break;
      default: return(false);
   }

   string filename = InpSymbol + "_" + tf_str + "_lookup.csv";

   // Try to open from Common folder first (Strategy Tester uses this)
   int file_handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(file_handle == INVALID_HANDLE)
   {
      // Try without FILE_COMMON flag (for regular Expert mode)
      file_handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
      if(file_handle == INVALID_HANDLE)
      {
         Print("ERROR: Cannot open ", filename, " Error: ", GetLastError());
         Print("       Tried both locations:");
         Print("       1. Terminal\\Common\\Files\\ (for Strategy Tester)");
         Print("       2. Terminal\\...\\MQL5\\Files\\ (for regular EA)");
         Print("       Copy files to: Terminal\\Common\\Files\\");
         return(false);
      }
   }

   // Read and check header line
   string header_line = "";
   while(!FileIsEnding(file_handle))
   {
      header_line = FileReadString(file_handle);
      if(header_line != "") break;
   }

   bool has_full_format =
      (StringFind(header_line, "change_pct")   >= 0 &&
       StringFind(header_line, "ensemble_std") >= 0);

   if(InpShowDebug)
   {
      if(has_full_format)
         Print("  → ", filename, " has full format (4 columns)");
      else
         Print("  → ", filename, " has simple format (2 columns) - will auto-calculate missing values");
   }

   // Read all records - parse line by line
   ArrayResize(temp_array, 10000); // Initial size
   double last_price = 0;

   while(!FileIsEnding(file_handle))
   {
      string line = FileReadString(file_handle);
      if(line == "" || StringLen(line) < 5) continue;

      string parts[];
      int    num_parts = StringSplit(line, ',', parts);

      if(num_parts < 2) continue; // Need at least timestamp and prediction

      // Parse timestamp (first column)
      string timestamp_str = parts[0];
      StringTrimLeft(timestamp_str);
      StringTrimRight(timestamp_str);

      StringReplace(timestamp_str, ".", "-"); // Convert dots to dashes
      datetime dt = StringToTime(timestamp_str);

      // Parse prediction (second column)
      double prediction = StringToDouble(parts[1]);

      double change_pct   = 0;
      double ensemble_std = 0.025; // Default uncertainty

      // If full format, read the additional columns
      if(has_full_format && num_parts >= 4)
      {
         change_pct   = StringToDouble(parts[2]);
         ensemble_std = StringToDouble(parts[3]);
      }
      else
      {
         // Calculate change_pct from prediction
         if(last_price > 0)
            change_pct = ((prediction - last_price) / last_price) * 100.0;
         else
            change_pct = 0.0;
      }

      if(dt > 0 && prediction > 0)
      {
         if(count >= ArraySize(temp_array))
            ArrayResize(temp_array, count + 1000);

         temp_array[count].timestamp    = dt;
         temp_array[count].prediction   = prediction;
         temp_array[count].change_pct   = change_pct;
         temp_array[count].ensemble_std = ensemble_std;
         count++;

         last_price = prediction;

         if(InpShowDebug && count <= 3)
         {
            Print("  Record ", count, ": Time=", TimeToString(dt),
                  " Pred=", prediction, " Change%=", change_pct);
         }
      }
   }

   FileClose(file_handle);

   if(count == 0)
   {
      Print("ERROR: No valid records found in ", filename);
      return(false);
   }

   ArrayResize(temp_array, count);

   switch(timeframe)
   {
      case PERIOD_H1:
         ArrayResize(g_csv_1H, count);
         ArrayCopy(g_csv_1H, temp_array);
         g_csv_1H_count = count;
         break;
      case PERIOD_H4:
         ArrayResize(g_csv_4H, count);
         ArrayCopy(g_csv_4H, temp_array);
         g_csv_4H_count = count;
         break;
      case PERIOD_D1:
         ArrayResize(g_csv_1D, count);
         ArrayCopy(g_csv_1D, temp_array);
         g_csv_1D_count = count;
         break;
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Lookup predictions from CSV based on timestamp                   |
//+------------------------------------------------------------------+
bool LookupPredictionsFromCSV(datetime lookup_time)
{
   bool found = false;

   // Lookup 1H prediction
   for(int i = 0; i < g_csv_1H_count; i++)
   {
      if(g_csv_1H[i].timestamp == lookup_time)
      {
         g_pred_1H.prediction   = g_csv_1H[i].prediction;
         g_pred_1H.change_pct   = g_csv_1H[i].change_pct;
         g_pred_1H.ensemble_std = g_csv_1H[i].ensemble_std;
         g_pred_1H.last_update  = lookup_time;
         found                  = true;
         break;
      }
   }

   // Lookup 4H prediction
   for(int i = 0; i < g_csv_4H_count; i++)
   {
      if(g_csv_4H[i].timestamp == lookup_time)
      {
         g_pred_4H.prediction   = g_csv_4H[i].prediction;
         g_pred_4H.change_pct   = g_csv_4H[i].change_pct;
         g_pred_4H.ensemble_std = g_csv_4H[i].ensemble_std;
         g_pred_4H.last_update  = lookup_time;
         found                  = true;
         break;
      }
   }

   // Lookup 1D prediction
   for(int i = 0; i < g_csv_1D_count; i++)
   {
      if(g_csv_1D[i].timestamp == lookup_time)
      {
         g_pred_1D.prediction   = g_csv_1D[i].prediction;
         g_pred_1D.change_pct   = g_csv_1D[i].change_pct;
         g_pred_1D.ensemble_std = g_csv_1D[i].ensemble_std;
         g_pred_1D.last_update  = lookup_time;
         found                  = true;
         break;
      }
   }

   if(found)
   {
      // Get current price
      g_current_price = SymbolInfoDouble(InpSymbol, SYMBOL_BID);

      // Update prediction records for accuracy tracking
      UpdatePredictionRecord(g_tracker_1H, g_pred_1H.prediction, PeriodSeconds(PERIOD_H1));
      UpdatePredictionRecord(g_tracker_4H, g_pred_4H.prediction, PeriodSeconds(PERIOD_H4));
      UpdatePredictionRecord(g_tracker_1D, g_pred_1D.prediction, PeriodSeconds(PERIOD_D1));

      if(InpShowDebug)
      {
         Print("✓ CSV Predictions for ", TimeToString(lookup_time));
         Print("  1H: ", g_pred_1H.prediction, " (", g_pred_1H.change_pct, "%)");
         Print("  4H: ", g_pred_4H.prediction, " (", g_pred_4H.change_pct, "%)");
         Print("  1D: ", g_pred_1D.prediction, " (", g_pred_1D.change_pct, "%)");
      }
   }

   return(found);
}

//+------------------------------------------------------------------+
//| Initialize accuracy tracker                                      |
//+------------------------------------------------------------------+
void InitAccuracyTracker(AccuracyTracker &tracker, string timeframe_name)
{
   tracker.total_predictions                 = 0;
   tracker.accurate_predictions              = 0;
   tracker.accuracy_percent                  = 0.0;
   tracker.current_prediction.timestamp      = 0;
   tracker.current_prediction.predicted_price = 0;
   tracker.current_prediction.start_price    = 0;
   tracker.current_prediction.checked        = false;
   tracker.current_prediction.accurate       = false;
   tracker.current_prediction.check_time     = 0;
   tracker.current_prediction.timeframe_name = timeframe_name;
}

//+------------------------------------------------------------------+
//| Read predictions from JSON file (Live mode only)                 |
//+------------------------------------------------------------------+
bool ReadPredictions()
{
   ResetLastError();

   int file_handle = FileOpen(g_predictionsFile, FILE_READ | FILE_TXT | FILE_ANSI);

   if(file_handle == INVALID_HANDLE)
      return(false);

   string file_content = "";
   while(!FileIsEnding(file_handle))
   {
      string line = FileReadString(file_handle);
      file_content += line;
   }
   FileClose(file_handle);

   if(StringLen(file_content) == 0)
      return(false);

   // Parse JSON for all timeframes
   g_pred_1H.prediction   = ParsePredictionValue(file_content, "1H", "prediction");
   g_pred_1H.change_pct   = ParsePredictionValue(file_content, "1H", "change_pct");
   g_pred_1H.ensemble_std = ParsePredictionValue(file_content, "1H", "ensemble_std");

   g_pred_4H.prediction   = ParsePredictionValue(file_content, "4H", "prediction");
   g_pred_4H.change_pct   = ParsePredictionValue(file_content, "4H", "change_pct");
   g_pred_4H.ensemble_std = ParsePredictionValue(file_content, "4H", "ensemble_std");

   g_pred_1D.prediction   = ParsePredictionValue(file_content, "1D", "prediction");
   g_pred_1D.change_pct   = ParsePredictionValue(file_content, "1D", "change_pct");
   g_pred_1D.ensemble_std = ParsePredictionValue(file_content, "1D", "ensemble_std");

   if(g_pred_1H.prediction > 0 && g_pred_4H.prediction > 0 && g_pred_1D.prediction > 0)
   {
      datetime now = TimeCurrent();
      g_pred_1H.last_update = now;
      g_pred_4H.last_update = now;
      g_pred_1D.last_update = now;

      g_current_price = SymbolInfoDouble(InpSymbol, SYMBOL_BID);

      UpdatePredictionRecord(g_tracker_1H, g_pred_1H.prediction, PeriodSeconds(PERIOD_H1));
      UpdatePredictionRecord(g_tracker_4H, g_pred_4H.prediction, PeriodSeconds(PERIOD_H4));
      UpdatePredictionRecord(g_tracker_1D, g_pred_1D.prediction, PeriodSeconds(PERIOD_D1));

      if(InpShowDebug)
      {
         Print("✓ Predictions updated:");
         Print("  1H: ", g_pred_1H.prediction, " (", g_pred_1H.change_pct, "%)");
         Print("  4H: ", g_pred_4H.prediction, " (", g_pred_4H.change_pct, "%)");
         Print("  1D: ", g_pred_1D.prediction, " (", g_pred_1D.change_pct, "%)");
      }

      return(true);
   }

   return(false);
}

//+------------------------------------------------------------------+
//| Parse prediction value from JSON string                          |
//+------------------------------------------------------------------+
double ParsePredictionValue(string json, string timeframe, string field)
{
   string search_pattern = "\"" + timeframe + "\"";
   int    start_pos      = StringFind(json, search_pattern);
   if(start_pos < 0) return(0.0);

   int field_pos = StringFind(json, "\"" + field + "\"", start_pos);
   if(field_pos < 0) return(0.0);

   int colon_pos = StringFind(json, ":", field_pos);
   if(colon_pos < 0) return(0.0);

   string after_colon = StringSubstr(json, colon_pos + 1);
   StringTrimLeft(after_colon);
   StringTrimRight(after_colon);

   string number_str = "";
   for(int i = 0; i < StringLen(after_colon); i++)
   {
      ushort ch = StringGetCharacter(after_colon, i);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == 'e' || ch == 'E')
      {
         number_str += ShortToString(ch);
      }
      else if(StringLen(number_str) > 0)
      {
         break;
      }
   }

   if(StringLen(number_str) == 0) return(0.0);

   return(StringToDouble(number_str));
}

//+------------------------------------------------------------------+
//| Update prediction record for accuracy tracking                   |
//+------------------------------------------------------------------+
void UpdatePredictionRecord(AccuracyTracker &tracker, double new_prediction, int period_seconds)
{
   if(MathAbs(new_prediction - tracker.current_prediction.predicted_price) > 0.00001 &&
      new_prediction > 0)
   {
      if(tracker.current_prediction.predicted_price > 0 &&
         !tracker.current_prediction.checked)
      {
         CheckPredictionAccuracy(tracker);
      }

      tracker.current_prediction.timestamp       = TimeCurrent();
      tracker.current_prediction.predicted_price = new_prediction;
      tracker.current_prediction.start_price     = g_current_price;
      tracker.current_prediction.checked         = false;
      tracker.current_prediction.accurate        = false;
      tracker.current_prediction.check_time      = tracker.current_prediction.timestamp + period_seconds;

      if(InpShowDebug)
      {
         Print("→ New ", tracker.current_prediction.timeframe_name, " prediction: ",
               new_prediction, " | Check at: ",
               TimeToString(tracker.current_prediction.check_time, TIME_DATE | TIME_MINUTES));
      }
   }
}

//+------------------------------------------------------------------+
//| Check accuracy for a timeframe                                   |
//+------------------------------------------------------------------+
void CheckAccuracy(AccuracyTracker &tracker, ENUM_TIMEFRAMES timeframe)
{
   if(!tracker.current_prediction.checked &&
      tracker.current_prediction.predicted_price > 0)
   {
      if(TimeCurrent() >= tracker.current_prediction.check_time)
         CheckPredictionAccuracy(tracker);
   }
}

//+------------------------------------------------------------------+
//| Check if prediction was accurate                                 |
//+------------------------------------------------------------------+
void CheckPredictionAccuracy(AccuracyTracker &tracker)
{
   if(tracker.current_prediction.checked) return;

   datetime start_time      = tracker.current_prediction.timestamp;
   datetime end_time        = tracker.current_prediction.check_time;
   double   predicted_price = tracker.current_prediction.predicted_price;
   double   start_price     = tracker.current_prediction.start_price;

   bool predicted_up = (predicted_price > start_price);

   MqlRates rates[];
   int      period_seconds = (int)(end_time - start_time);
   int      bars           = (int)(period_seconds / PeriodSeconds(PERIOD_M1)) + 10;
   int      copied         = CopyRates(InpSymbol, PERIOD_M1, start_time, bars, rates);

   if(copied > 0)
   {
      bool price_reached = false;

      for(int i = 0; i < copied; i++)
      {
         if(rates[i].time > end_time) break;

         if(predicted_up)
         {
            if(rates[i].high >= predicted_price)
            {
               price_reached = true;
               break;
            }
         }
         else
         {
            if(rates[i].low <= predicted_price)
            {
               price_reached = true;
               break;
            }
         }
      }

      tracker.current_prediction.accurate = price_reached;
      tracker.current_prediction.checked  = true;

      tracker.total_predictions++;
      if(price_reached)
         tracker.accurate_predictions++;

      if(tracker.total_predictions > 0)
         tracker.accuracy_percent =
            (double)tracker.accurate_predictions / tracker.total_predictions * 100.0;

      Print("✓✓ ", tracker.current_prediction.timeframe_name, " Accuracy Check ✓✓");
      Print("   Direction: ", (predicted_up ? "UP ↑" : "DOWN ↓"));
      Print("   Predicted: ", DoubleToString(predicted_price, _Digits));
      Print("   Result: ", (price_reached ? "✓ ACCURATE" : "✗ INACCURATE"));
      Print("   Stats: ", tracker.accurate_predictions, "/", tracker.total_predictions,
            " = ", DoubleToString(tracker.accuracy_percent, 1), "%");
   }
}

//+------------------------------------------------------------------+
//| Check if current day is a trading day                            |
//+------------------------------------------------------------------+
bool IsTradingDay()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   switch(dt.day_of_week)
   {
      case 0: return InpTradeSunday;    // Sunday
      case 1: return InpTradeMonday;    // Monday
      case 2: return InpTradeTuesday;   // Tuesday
      case 3: return InpTradeWednesday; // Wednesday
      case 4: return InpTradeThursday;  // Thursday
      case 5: return InpTradeFriday;    // Friday
      case 6: return InpTradeSaturday;  // Saturday
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check if current time is within trading sessions                 |
//+------------------------------------------------------------------+
bool IsWithinTradingHours()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   int currentMinutes = dt.hour * 60 + dt.min;
   
   // Check Session 1
   if(InpUseSession1)
   {
      int session1Start = InpSession1StartHour * 60 + InpSession1StartMinute;
      int session1End = InpSession1EndHour * 60 + InpSession1EndMinute;
      
      // Handle sessions that cross midnight
      if(session1End < session1Start)
      {
         if(currentMinutes >= session1Start || currentMinutes <= session1End)
            return true;
      }
      else
      {
         if(currentMinutes >= session1Start && currentMinutes <= session1End)
            return true;
      }
   }
   
   // Check Session 2
   if(InpUseSession2)
   {
      int session2Start = InpSession2StartHour * 60 + InpSession2StartMinute;
      int session2End = InpSession2EndHour * 60 + InpSession2EndMinute;
      
      if(session2End < session2Start)
      {
         if(currentMinutes >= session2Start || currentMinutes <= session2End)
            return true;
      }
      else
      {
         if(currentMinutes >= session2Start && currentMinutes <= session2End)
            return true;
      }
   }
   
   // Check Session 3
   if(InpUseSession3)
   {
      int session3Start = InpSession3StartHour * 60 + InpSession3StartMinute;
      int session3End = InpSession3EndHour * 60 + InpSession3EndMinute;
      
      if(session3End < session3Start)
      {
         if(currentMinutes >= session3Start || currentMinutes <= session3End)
            return true;
      }
      else
      {
         if(currentMinutes >= session3Start && currentMinutes <= session3End)
            return true;
      }
   }
   
   // If no session is enabled or time is outside all sessions
   return (!InpUseSession1 && !InpUseSession2 && !InpUseSession3);
}

//+------------------------------------------------------------------+
//| Manage trailing stop for open positions                          |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   if(!InpUseTrailingStop)
      return;
      
   if(!PositionSelect(InpSymbol))
      return;
   
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   double pip = point;
   if(_Digits == 3 || _Digits == 5)
      pip = point * 10.0;
   
   double trailingDistance = InpTrailingStopPips * pip;
   double trailingStep = InpTrailingStepPips * pip;
   
   ulong ticket = PositionGetInteger(POSITION_TICKET);
   long posType = PositionGetInteger(POSITION_TYPE);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentSL = PositionGetDouble(POSITION_SL);
   double currentTP = PositionGetDouble(POSITION_TP);
   
   double bid = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
   
   // For BUY positions
   if(posType == POSITION_TYPE_BUY)
   {
      double profit = bid - openPrice;
      
      // Only activate trailing stop when in profit by the trailing stop distance
      if(profit >= trailingDistance)
      {
         double newSL = bid - trailingDistance;
         
         // Only modify if new SL is higher than current SL (or no SL exists)
         // and the price has moved enough to justify modification
         if((currentSL == 0 || newSL > currentSL + trailingStep))
         {
            if(g_trade.PositionModify(ticket, newSL, currentTP))
            {
               if(InpShowDebug)
                  Print("✓ Trailing stop updated for BUY position: New SL = ", newSL);
            }
         }
      }
   }
   // For SELL positions
   else if(posType == POSITION_TYPE_SELL)
   {
      double profit = openPrice - ask;
      
      // Only activate trailing stop when in profit by the trailing stop distance
      if(profit >= trailingDistance)
      {
         double newSL = ask + trailingDistance;
         
         // Only modify if new SL is lower than current SL (or no SL exists)
         // and the price has moved enough to justify modification
         if((currentSL == 0 || newSL < currentSL - trailingStep))
         {
            if(g_trade.PositionModify(ticket, newSL, currentTP))
            {
               if(InpShowDebug)
                  Print("✓ Trailing stop updated for SELL position: New SL = ", newSL);
            }
         }
      }
   }
}


//+------------------------------------------------------------------+
//| Check for trading signal                                         |
//+------------------------------------------------------------------+
void CheckForTradingSignal()
{
   // Check if trading is allowed on current day
   if(!IsTradingDay())
   {
      if(InpShowDebug)
         Print("Trading not allowed on this day");
      return;
   }
   
   // Check if within trading hours
   if(!IsWithinTradingHours())
   {
      if(InpShowDebug)
         Print("Outside trading hours");
      return;
   }

   // Don't trade too frequently
   if(TimeCurrent() - g_last_trade_time < g_min_trade_interval)
      return;

   // Check if we already have an open position
   if(PositionSelect(InpSymbol))
      return;

   // Get the prediction for the selected trading timeframe
   PredictionData selected_pred;
   string         tf_name = "";

   switch(InpTradingTimeframe)
   {
      case PERIOD_H1:
         selected_pred = g_pred_1H;
         tf_name       = "1H";
         break;
      case PERIOD_H4:
         selected_pred = g_pred_4H;
         tf_name       = "4H";
         break;
      case PERIOD_D1:
         selected_pred = g_pred_1D;
         tf_name       = "1D";
         break;
      default:
         Print("ERROR: Unsupported trading timeframe");
         return;
   }

   if(selected_pred.prediction <= 0)
      return;

   // Determine pip size
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   double pip   = point;
   if(_Digits == 3 || _Digits == 5)
      pip = point * 10.0;

   // Compute prediction distance in pips
   double delta_pips = (selected_pred.prediction - g_current_price) / pip;

   bool signal_buy  = false;
   bool signal_sell = false;

   if(InpMinPredictionPips <= 0)
   {
      // Old behavior: any prediction above/below price
      signal_buy  = (selected_pred.prediction > g_current_price);
      signal_sell = (selected_pred.prediction < g_current_price);
   }
   else
   {
      // New behavior: require minimum prediction distance in pips
      signal_buy  = (delta_pips >=  InpMinPredictionPips);
      signal_sell = (delta_pips <= -InpMinPredictionPips);

      if(InpShowDebug)
      {
         Print("Prediction delta: ", DoubleToString(delta_pips, 1),
               " pips | Min required: ", InpMinPredictionPips);
      }
   }

   // If no valid direction, exit
   if(!signal_buy && !signal_sell)
   {
      if(InpShowDebug)
         Print("No trade: prediction distance not sufficient or no clear direction.");
      return;
   }

   // Apply trend filter
   if(InpUseTrendFilter)
   {
      if(!CheckTrendFilter(signal_buy, signal_sell))
      {
         if(InpShowDebug)
            Print("Trade rejected by trend filter");
         return;
      }
   }

   // Apply RSI filter
   if(InpUseRSIFilter)
   {
      if(!CheckRSIFilter(signal_buy, signal_sell))
      {
         if(InpShowDebug)
            Print("Trade rejected by RSI filter");
         return;
      }
   }

   // Calculate position size
   double lot_size = CalculateLotSize();
   if(lot_size <= 0)
      return;

   // Calculate SL and TP
   double sl_distance = InpStopLossPips * pip;
   double tp_price    = 0;
   double tp_distance = 0;

   // Use predicted price as TP or fixed pips
   if(InpUsePredictedPrice)
   {
      // Use the predicted price as take profit target
      tp_price = selected_pred.prediction * InpTPMultiplier;

      // Validate TP is reasonable distance from current price
      double tp_pips = MathAbs(tp_price - g_current_price) / pip;

      if(tp_pips < InpMinTPPips)
      {
         if(InpShowDebug)
            Print("TP too close (", tp_pips, " pips). Minimum is ",
                  InpMinTPPips, " pips.");
         return;
      }

      if(tp_pips > InpMaxTPPips)
      {
         if(InpShowDebug)
            Print("TP too far (", tp_pips, " pips). Maximum is ",
                  InpMaxTPPips, " pips. Capping.");

         // Cap at maximum
         if(signal_buy)
            tp_price = g_current_price + (InpMaxTPPips * pip);
         else
            tp_price = g_current_price - (InpMaxTPPips * pip);
      }

      if(InpShowDebug)
         Print("Using predicted price as TP: ", tp_price, " (",
               tp_pips, " pips)");
   }
   else
   {
      // Use fixed pips TP
      tp_distance = InpTakeProfitPips * pip;
   }

   // Execute trade
   if(signal_buy)
   {
      double ask = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
      double sl  = ask - sl_distance;
      double tp  = InpUsePredictedPrice ? tp_price : ask + tp_distance;

      // Validate TP is above entry for buy
      if(tp <= ask)
      {
         if(InpShowDebug)
            Print("Invalid BUY TP: ", tp, " must be above entry: ", ask);
         return;
      }

      string comment = StringFormat("ML EA v7 [%s] %s%.2f%% → TP:%.5f",
                                    tf_name,
                                    (selected_pred.change_pct >= 0 ? "+" : ""),
                                    selected_pred.change_pct,
                                    tp);

      if(g_trade.Buy(lot_size, InpSymbol, ask, sl, tp, comment))
      {
         g_last_trade_time = TimeCurrent();
         Print("✓ BUY order placed - ", tf_name,
               " prediction: ", selected_pred.prediction,
               " | TP: ", tp, " | SL: ", sl,
               " | Δ=", DoubleToString(delta_pips, 1), " pips");
      }
   }
   else if(signal_sell)
   {
      double bid = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
      double sl  = bid + sl_distance;
      double tp  = InpUsePredictedPrice ? tp_price : bid - tp_distance;

      // Validate TP is below entry for sell
      if(tp >= bid)
      {
         if(InpShowDebug)
            Print("Invalid SELL TP: ", tp, " must be below entry: ", bid);
         return;
      }

      string comment = StringFormat("ML EA v7 [%s] %.2f%% → TP:%.5f",
                                    tf_name,
                                    selected_pred.change_pct,
                                    tp);

      if(g_trade.Sell(lot_size, InpSymbol, bid, sl, tp, comment))
      {
         g_last_trade_time = TimeCurrent();
         Print("✓ SELL order placed - ", tf_name,
               " prediction: ", selected_pred.prediction,
               " | TP: ", tp, " | SL: ", sl,
               " | Δ=", DoubleToString(delta_pips, 1), " pips");
      }
   }
}

//+------------------------------------------------------------------+
//| Check trend filter                                               |
//+------------------------------------------------------------------+
bool CheckTrendFilter(bool &signal_buy, bool &signal_sell)
{
   double ma_buffer[];
   ArraySetAsSeries(ma_buffer, true);

   if(CopyBuffer(g_handle_trend_ma, 0, 0, 2, ma_buffer) != 2)
      return(false);

   double current_ma    = ma_buffer[0];
   double current_close = iClose(InpSymbol, Period(), 0);

   // Only allow buy if price is above MA
   if(signal_buy && current_close < current_ma)
   {
      signal_buy = false;
      return(false);
   }

   // Only allow sell if price is below MA
   if(signal_sell && current_close > current_ma)
   {
      signal_sell = false;
      return(false);
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Check RSI filter                                                 |
//+------------------------------------------------------------------+
bool CheckRSIFilter(bool &signal_buy, bool &signal_sell)
{
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);

   if(CopyBuffer(g_handle_rsi, 0, 0, 1, rsi_buffer) != 1)
      return(false);

   double current_rsi = rsi_buffer[0];

   // Don't buy if RSI is overbought
   if(signal_buy && current_rsi > InpRSIOverbought)
   {
      signal_buy = false;
      return(false);
   }

   // Don't sell if RSI is oversold
   if(signal_sell && current_rsi < InpRSIOversold)
   {
      signal_sell = false;
      return(false);
   }

   return(true);
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                 |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = balance * (InpRiskPercent / 100.0);

   double point     = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   double pip       = point;
   if(_Digits == 3 || _Digits == 5)
      pip = point * 10.0;

   double tick_size  = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_VALUE);

   if(tick_size <= 0 || tick_value <= 0)
   {
      Print("ERROR: Invalid tick_size or tick_value, cannot calculate lot size.");
      return(0.0);
   }

   double sl_price_distance = InpStopLossPips * pip;       // price units
   double sl_ticks          = sl_price_distance / tick_size; // ticks

   if(sl_ticks <= 0)
   {
      Print("ERROR: SL ticks <= 0, cannot calculate lot size.");
      return(0.0);
   }

   double risk_per_lot = sl_ticks * tick_value; // currency per 1 lot at SL

   if(risk_per_lot <= 0)
   {
      Print("ERROR: risk_per_lot <= 0, cannot calculate lot size.");
      return(0.0);
   }

   double lot_size = risk_amount / risk_per_lot;

   // Normalize lot size
   double min_lot  = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   double max_lot  = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);

   if(lot_step <= 0)
      lot_step = min_lot;

   lot_size = MathFloor(lot_size / lot_step) * lot_step;
   lot_size = MathMax(lot_size, min_lot);
   lot_size = MathMin(lot_size, max_lot);

   return(lot_size);
}

//+------------------------------------------------------------------+
//| Update display                                                   |
//+------------------------------------------------------------------+
void UpdateDisplay()
{
   int x_pos      = InpXOffset;
   int y_pos      = InpYOffset;
   int line_height = InpFontSize + 8;

   // Header
   CreateLabel("MLEA_Header", x_pos, y_pos,
               "╔══════════════════════════════════════╗",
               InpFontSize, InpTextColor);
   y_pos += line_height;

   string mode_indicator = InpStrategyTesterMode ? "[TESTER]" : "[LIVE]";
   CreateLabel("MLEA_Title", x_pos, y_pos,
               "║   ML PREDICTIONS " + mode_indicator + " v7.11   ║",
               InpFontSize, clrYellow);
   y_pos += line_height;

   CreateLabel("MLEA_Separator1", x_pos, y_pos,
               "╠══════════════════════════════════════╣",
               InpFontSize, InpTextColor);
   y_pos += line_height;

   // Current price
   string current_text =
      StringFormat("║ Current: %." + IntegerToString(_Digits) + "f", g_current_price);
   while(StringLen(current_text) < 38) current_text += " ";
   current_text += "║";
   CreateLabel("MLEA_Current", x_pos, y_pos, current_text, InpFontSize, clrAqua);
   y_pos += line_height;

   CreateLabel("MLEA_Separator2", x_pos, y_pos,
               "╠══════════════════════════════════════╣",
               InpFontSize, InpTextColor);
   y_pos += line_height;

   // 1H Prediction
   DisplayPredictionLine("1H", g_pred_1H, g_tracker_1H, x_pos, y_pos);
   y_pos += line_height;

   // 4H Prediction
   DisplayPredictionLine("4H", g_pred_4H, g_tracker_4H, x_pos, y_pos);
   y_pos += line_height;

   // 1D Prediction
   DisplayPredictionLine("1D", g_pred_1D, g_tracker_1D, x_pos, y_pos);
   y_pos += line_height;

   CreateLabel("MLEA_Separator3", x_pos, y_pos,
               "╠══════════════════════════════════════╣",
               InpFontSize, InpTextColor);
   y_pos += line_height;

   // Overall accuracy
   int    total_all    = g_tracker_1H.total_predictions +
                         g_tracker_4H.total_predictions +
                         g_tracker_1D.total_predictions;
   int    accurate_all = g_tracker_1H.accurate_predictions +
                         g_tracker_4H.accurate_predictions +
                         g_tracker_1D.accurate_predictions;
   double accuracy_all = (total_all > 0)
                         ? (double)accurate_all / total_all * 100.0
                         : 0.0;

   string overall_text =
      StringFormat("║ Overall: %d/%d (%.1f%%)              ",
                   accurate_all, total_all, accuracy_all);
   while(StringLen(overall_text) < 38) overall_text += " ";
   overall_text += "║";
   CreateLabel("MLEA_Overall", x_pos, y_pos, overall_text, InpFontSize, clrGold);
   y_pos += line_height;

   // Trading status
   string trading_tf = "";
   switch(InpTradingTimeframe)
   {
      case PERIOD_H1: trading_tf = "1H"; break;
      case PERIOD_H4: trading_tf = "4H"; break;
      case PERIOD_D1: trading_tf = "1D"; break;
   }

   string status_text =
      StringFormat("║ Trading: %s [%s]                 ",
                   (InpEnableTrading ? "ON" : "OFF"), trading_tf);
   while(StringLen(status_text) < 38) status_text += " ";
   status_text += "║";
   CreateLabel("MLEA_Status", x_pos, y_pos, status_text,
               InpFontSize - 1, InpEnableTrading ? clrLimeGreen : clrOrange);
   y_pos += line_height;

   CreateLabel("MLEA_Footer", x_pos, y_pos,
               "╚══════════════════════════════════════╝",
               InpFontSize, InpTextColor);
   y_pos += line_height;

   // Show active filters and settings
   string filters = "Filters: ";
   if(InpUseTrendFilter)
      filters += "Trend(" + IntegerToString(InpTrendMAPeriod) + ") ";
   if(InpUseRSIFilter)
      filters += "RSI(" + IntegerToString(InpRSIPeriod) + ") ";
   if(!InpUseTrendFilter && !InpUseRSIFilter)
      filters += "None";

   CreateLabel("MLEA_Filters", x_pos, y_pos, filters, InpFontSize - 1, clrGray);
   y_pos += line_height - 3;

   // Show TP mode
   string tp_mode = "TP: ";
   if(InpUsePredictedPrice)
      tp_mode += "Predicted Price (x" + DoubleToString(InpTPMultiplier, 1) + ")";
   else
      tp_mode += "Fixed " + IntegerToString(InpTakeProfitPips) + " pips";

   CreateLabel("MLEA_TPMode", x_pos, y_pos, tp_mode, InpFontSize - 1, clrGray);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Display prediction line                                          |
//+------------------------------------------------------------------+
void DisplayPredictionLine(string tf_name, PredictionData &pred,
                           AccuracyTracker &tracker, int x_pos, int &y_pos)
{
   color line_color = (pred.prediction > g_current_price) ? InpUpColor : InpDownColor;
   string arrow     = (pred.prediction > g_current_price) ? "↑" : "↓";

   string pred_text =
      StringFormat("║ %s: %." + IntegerToString(_Digits) + "f %s %.2f%%",
                   tf_name, pred.prediction, arrow, pred.change_pct);
   while(StringLen(pred_text) < 38) pred_text += " ";
   pred_text += "║";

   CreateLabel("MLEA_Pred_" + tf_name, x_pos, y_pos, pred_text,
               InpFontSize, line_color);
   y_pos += InpFontSize + 5;

   string accuracy_text =
      StringFormat("║    Accuracy: %d/%d (%.1f%%)          ",
                   tracker.accurate_predictions,
                   tracker.total_predictions,
                   tracker.accuracy_percent);
   while(StringLen(accuracy_text) < 38) accuracy_text += " ";
   accuracy_text += "║";

   color acc_color = (tracker.accuracy_percent >= 60) ? clrLimeGreen :
                     (tracker.accuracy_percent >= 50) ? clrYellow : clrRed;

   CreateLabel("MLEA_Acc_" + tf_name, x_pos, y_pos, accuracy_text,
               InpFontSize - 1, acc_color);
}

//+------------------------------------------------------------------+
//| Display error message                                            |
//+------------------------------------------------------------------+
void DisplayError()
{
   int x_pos      = InpXOffset;
   int y_pos      = InpYOffset;
   int line_height = InpFontSize + 8;

   CreateLabel("MLEA_Error1", x_pos, y_pos,
               "⚠ Waiting for ML predictions...",
               InpFontSize + 2, clrOrange);
   y_pos += line_height + 5;

   CreateLabel("MLEA_Error2", x_pos, y_pos,
               "Run: python predictor-7.py predict --symbol " + InpSymbol,
               InpFontSize - 1, clrGray);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Create label helper                                              |
//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text,
                 int font_size, color clr)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   }

   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, font_size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetString(0, name, OBJPROP_FONT, "Courier New");
}

//+------------------------------------------------------------------+
//| Save accuracy data                                               |
//+------------------------------------------------------------------+
void SaveAccuracyData()
{
   string filename = "accuracy_" + InpSymbol + "_v7.dat";
   int    handle   = FileOpen(filename, FILE_WRITE | FILE_BIN);

   if(handle != INVALID_HANDLE)
   {
      FileWriteInteger(handle, g_tracker_1H.total_predictions);
      FileWriteInteger(handle, g_tracker_1H.accurate_predictions);
      FileWriteInteger(handle, g_tracker_4H.total_predictions);
      FileWriteInteger(handle, g_tracker_4H.accurate_predictions);
      FileWriteInteger(handle, g_tracker_1D.total_predictions);
      FileWriteInteger(handle, g_tracker_1D.accurate_predictions);
      FileClose(handle);

      if(InpShowDebug)
         Print("✓ Accuracy data saved");
   }
}

//+------------------------------------------------------------------+
//| Load accuracy data                                               |
//+------------------------------------------------------------------+
void LoadAccuracyData()
{
   string filename = "accuracy_" + InpSymbol + "_v7.dat";
   int    handle   = FileOpen(filename, FILE_READ | FILE_BIN);

   if(handle != INVALID_HANDLE)
   {
      g_tracker_1H.total_predictions      = FileReadInteger(handle);
      g_tracker_1H.accurate_predictions   = FileReadInteger(handle);
      if(g_tracker_1H.total_predictions > 0)
         g_tracker_1H.accuracy_percent =
            (double)g_tracker_1H.accurate_predictions /
            g_tracker_1H.total_predictions * 100.0;

      g_tracker_4H.total_predictions      = FileReadInteger(handle);
      g_tracker_4H.accurate_predictions   = FileReadInteger(handle);
      if(g_tracker_4H.total_predictions > 0)
         g_tracker_4H.accuracy_percent =
            (double)g_tracker_4H.accurate_predictions /
            g_tracker_4H.total_predictions * 100.0;

      g_tracker_1D.total_predictions      = FileReadInteger(handle);
      g_tracker_1D.accurate_predictions   = FileReadInteger(handle);
      if(g_tracker_1D.total_predictions > 0)
         g_tracker_1D.accuracy_percent =
            (double)g_tracker_1D.accurate_predictions /
            g_tracker_1D.total_predictions * 100.0;

      FileClose(handle);
      Print("✓ Loaded historical accuracy data");
   }
}
//+------------------------------------------------------------------+
