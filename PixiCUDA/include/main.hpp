#pragma once

#ifndef _MAIN_H_
#define _MAIN_H_

static void onTrackbarAngle(const int angle, void* userdata);
static void onTrackbarDistance(const int distance, void* userdata);
static void onTrackbarAlgoSelection(const int selection, void* userdata);
static void onTrackbarCheckPrecision(const int state, void* userdata);
static void onTrackbarThreadsBinLog(const int threads, void* userdata);

#endif // !_MAIN_H_
