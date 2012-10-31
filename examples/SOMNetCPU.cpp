/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>
#include <math/ANRandom.h>
#include <gui/QSOMReader.h>

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  std::vector<float> red, green, blue, yellow, orange, purple, dk_green, dk_blue, black, white;

  white.push_back(1);
  white.push_back(1);
  white.push_back(1);

  black.push_back(0);
  black.push_back(0);
  black.push_back(0);

  red.push_back(1);
  red.push_back(0);
  red.push_back(0);

  green.push_back(0);
  green.push_back(1);
  green.push_back(0);

  dk_green.push_back(0);
  dk_green.push_back(0.5);
  dk_green.push_back(0.25);

  blue.push_back(0);
  blue.push_back(0);
  blue.push_back(1);

  dk_blue.push_back(0);
  dk_blue.push_back(0);
  dk_blue.push_back(0.5);

  yellow.push_back(1);
  yellow.push_back(1);
  yellow.push_back(0.2);

  orange.push_back(1);
  orange.push_back(0.4);
  orange.push_back(0.25);

  purple.push_back(1);
  purple.push_back(0);
  purple.push_back(1);

  ANN::TrainingSet input;
  input.AddInput(red);
  input.AddInput(green);
  input.AddInput(dk_green);
  input.AddInput(blue);
  input.AddInput(dk_blue);
  input.AddInput(yellow);
  input.AddInput(orange);
  input.AddInput(purple);
  input.AddInput(black);
  input.AddInput(white);

  std::vector<float> vCol(3);
  int w1 = 40;
  int w2 = 8;

  ANN::SOMNet SOMap;
  SOMap.SetTrainingSet(input);
  SOMap.CreateSOM(3, 1, w1,w1);

  SOMap.Training(400);

  SOMReader w(w1, w1, w2);
  for(int x = 0; x < w1*w1; x++) {
  	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)SOMap.GetOPLayer())->GetNeuron(x);
  	vCol[0] = pNeur->GetConI(0)->GetValue();
  	vCol[1] = pNeur->GetConI(1)->GetValue();
  	vCol[2] = pNeur->GetConI(2)->GetValue();

  	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
  }
  w.Save("CPU.png");

  return 0;
}
