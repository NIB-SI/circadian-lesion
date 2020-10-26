#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <algorithm> // std::min_element
#include <iterator>

#include <math.h>       /* acos */
#define PI 3.14159265

using namespace cv;
using namespace std;

Mat normalization(Mat adjusted_brightness)
{
	/*
	 * Hengen, Heiko, Susanne L. Spoor, and Madhukar C. Pandit.
	 * "Analysis of blood and bone marrow smears using digital image processing techniques."
	 * Proc. SPIE. Vol. 4684. 2002.
	 */

	Mat hsv;
	cvtColor(adjusted_brightness,hsv,COLOR_BGR2HSV);

	vector<cv::Mat> channel;
	split(hsv,channel); 

	int histSize = 255;

	float range[] = { 0, 255 } ; //the upper boundary is exclusive
	const float* histRange = { range };

	bool uniform = true; 
	bool accumulate = false;

	Mat i_hist;
	
	calcHist( &channel[2], 1, 0, Mat(), i_hist, 1, &histSize, &histRange, uniform, accumulate );

	Point max, min;
	minMaxLoc(i_hist, 0, 0, &min, &max);

	int minimum=1000; int min_index=0;

	for( int i = 1; i < histSize; i++ )
	{
		if(i_hist.at<float>(i)<minimum && i_hist.at<float>(i)!=0 && i<max.y){minimum=i_hist.at<float>(i); min_index=i;};
	}

	for( int i = 0; i < hsv.rows; i++ )
	{
		for( int j = 0; j < hsv.cols; j++ )
		{
			//cout<<"alt:"<<(int)hsv.at<Vec3b>(i, j)[2]<<endl;
			int value=(((double)hsv.at<Vec3b>(i, j)[2]-min_index)/(max.y-min_index))*(255-min_index)+min_index;
			if(value>255){value=255;}
			hsv.at<Vec3b>(i, j)[2]=value;
			//cout<<"neu:"<<int(hsv.at<Vec3b>(i, j)[2])<<endl;
		}
	}

	Mat normalized_input;
	cvtColor(hsv,normalized_input,COLOR_HSV2BGR);

	//imwrite("contrast.jpg", normalized_input);

	return normalized_input;
}
Mat findLesions1(Mat contrast, Mat input) //night images
{
	Mat hsv;
	cvtColor(contrast,hsv,COLOR_BGR2HSV);

	cv::Mat hue_range; //HSV color picker: http://pinetools.com/image-color-picker
	cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(50, 255, 200), hue_range);	

	copyMakeBorder(hue_range,hue_range,1,1,1,1,BORDER_CONSTANT,Scalar(0));

	Mat dil;
	dilate(hue_range, dil, Mat(), cv::Point(-1,-1), 7);
	erode(dil, dil, Mat(), cv::Point(-1,-1), 1);

	return dil;
}
Mat findLesions2(Mat contrast, Mat input) //day images
{
	Mat hsv;
	cvtColor(contrast,hsv,COLOR_BGR2HSV);

	cv::Mat hue_range, hue_range1, hue_range2; //HSV color picker: http://pinetools.com/image-color-picker
	cv::inRange(hsv, cv::Scalar(0, 100, 40), cv::Scalar(25, 255, 220), hue_range1);
	cv::inRange(hsv, cv::Scalar(175, 20, 40), cv::Scalar(180, 255, 255), hue_range2);

	//cv::addWeighted(hue_range1, 1.0, hue_range2, 1.0, 0.0, hue_range);

	copyMakeBorder(hue_range1,hue_range1,1,1,1,1,BORDER_CONSTANT,Scalar(0));

	Mat dil;
	dilate(hue_range1, dil, Mat(), cv::Point(-1,-1), 7);
	erode(dil, dil, Mat(), cv::Point(-1,-1), 3);

	return dil;
}

vector<int> measure_brightness(Mat input)
{
	int brightness=0;

	int blue=0;
	int red=0;
	int green=0;

	vector<int> intensity;

	for( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{		
			brightness+=(int)input.at<Vec3b>(i, j)[0]+(int)input.at<Vec3b>(i, j)[1]+(int)input.at<Vec3b>(i, j)[2];
			blue+=(int)input.at<Vec3b>(i, j)[0];	
			green+=(int)input.at<Vec3b>(i, j)[1];
			red+=(int)input.at<Vec3b>(i, j)[2];
		}
	}
	
	brightness=brightness/(3*input.rows*input.cols);
	blue=blue/(input.rows*input.cols);
	green=green/(input.rows*input.cols);
	red=red/(input.rows*input.cols);

	intensity.push_back(brightness);
	intensity.push_back(blue);
	intensity.push_back(green);
	intensity.push_back(red);

	return intensity;
}

Mat adjust_brightness(Mat input, vector<vector<int>> total_brightness, int c, int actual_image)
{
	Mat adjusted_brightness=input.clone();	

	for( int i = 0; i < adjusted_brightness.rows; i++ )
	{
		for( int j = 0; j < adjusted_brightness.cols; j++ )
		{	
			adjusted_brightness.at<Vec3b>(i, j)[0]+=(total_brightness[c][0]-total_brightness[actual_image][0]);
			//cout<<(int)input.at<Vec3b>(i, j)[0]<<endl;
			adjusted_brightness.at<Vec3b>(i, j)[1]+=(total_brightness[c][0]-total_brightness[actual_image][0]);
			adjusted_brightness.at<Vec3b>(i, j)[2]+=(total_brightness[c][0]-total_brightness[actual_image][0]);
		}
	}	
	cout<<total_brightness[c][0]<<endl;	

	return adjusted_brightness;
}

double area_intersecting_circles(Point2f m1, Point2f m2, float r1, float r2)
{
	int d=norm(m1-m2);
	double area;

	if(d<(r1+r2))
	{	
		if(d<=abs(r1-r2))
		{
			if(r1<r2){area=PI*pow(r1,2);}
			else{area=PI*pow(r2,2);}
		}

		else
		{
			double alpha=acos((pow(r2,2)+pow(d,2)-pow(r1,2))/(2*r2*d));
			double beta=acos((pow(r1,2)+pow(d,2)-pow(r2,2))/(2*r1*d));	
	
			area=beta*pow(r1,2)+alpha*pow(r2,2)-0.5*pow(r2,2)*sin(2*alpha)-0.5*pow(r1,2)*sin(2*beta);
		}
	
	}
	else{area=0;}

	return area;
}

int main( int argc, char** argv )
{
	String folder = argv[1];	
	vector<String> filenames;     	
  	  
	size_t last = folder.find_last_of('R');
	string experiment = folder.substr(last, string::npos); //Rywal bsp (point zero 2017_04_14 at 08.00.00)

	glob(folder, filenames);

	last=experiment.find_last_of('(');
	string start_time = experiment.substr(last+12, 22); //2017_04_14 at 08.00.00

	const char *c = start_time.c_str();
	struct tm tm;
	strptime(c, "%Y_%m_%d%nat%n%H.%M.%S", &tm);
	time_t timestamp = mktime(&tm); 

    	vector<int> timestamps, taf;
	vector<string> timepoints;
	timestamps.push_back(timestamp+3600);	
	//taf.push_back(0);

	Mat adjusted_brightness;
	vector<vector<int>> total_brightness;	
	int last_bright_image=0;

	vector<int>  lesionAreas;	
	vector<vector<int> > lesion;
	
	vector<Point>  mcs;
	vector<vector<Point>>  mp;

	vector<Rect> boundRects;
	vector<vector<Rect>> allboundRects;

	vector<float> radii;
	vector<vector<float>> allRadii;

	vector<Point2f> centers;
	vector<vector<Point2f>> allCenters;

	vector<int> numbers;
	vector<vector<int>> count;

    	for(size_t i = 0; i<filenames.size(); ++i) //filenames.size()
    	{  	
		cout<<i<<" ";
        	Mat input = imread(filenames[i]);

        	if(!input.data)
            	cerr << "Problem loading image!!!" << endl;

		last = filenames[i].find_last_of('-');
	  	string timepoint = filenames[i].substr(last+2, 15); //20170416_084124
		timepoints.push_back(timepoint);

		const char *c = timepoint.c_str();
		struct tm tm;
		strptime(c, "%Y%m%d_%H%M%S", &tm);
		time_t timestamp = mktime(&tm);

		timestamps.push_back(timestamp+3600); //1492328484
		
		if(i==0){taf.push_back((timestamps[i+1]-timestamps[i])/60);}
		else{taf.push_back((timestamps[i+1]-timestamps[i])/60+taf[i-1]);}		

		last = filenames[i].find_last_of('/');
	  	string filename = filenames[i].substr(last+1, string::npos);
	
		total_brightness.push_back(measure_brightness(input));

		if(total_brightness[i][0]<=100)
		{
			Mat contrast=normalization(input);
			Mat dil=findLesions1(contrast, input);

			//find contours of interest
			vector<vector<Point> > contours1;
			findContours(dil, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			vector<Moments> mu(contours1.size() );
			Point mc;

			Rect boundRect;
			float radius;
			Point2f center;

			//for each contour
			for ( size_t j = 0; j < contours1.size(); j++)
			{
				mu[j] = moments( contours1[j], false ); 
				mc = Point( mu[j].m10/mu[j].m00 , mu[j].m01/mu[j].m00 );

				//for contours with too small area
				if( contourArea(contours1[j]) < 300 || (mc.x-10)<0 || (mc.y-10)<0 || (mc.x+10)>input.cols || (mc.y+10)>input.rows)
				{
					//contours1.erase(contours1.begin() +j); 
					//mc.erase(mc.begin() +j);
					//j=j-1;
				}
				else
				{
					minEnclosingCircle( contours1[j], center, radius );
			
					if(radius<250)
					{
						drawContours(input, contours1, j, cv::Scalar(0, 0, 255),1, 8);
						boundRect = boundingRect( Mat(contours1[j]) );
						//rectangle(input, boundRect.tl(), boundRect.br(), Scalar(0, 0, 255),2,8,0); 				

										
						circle( input, center, radius, Scalar(0, 0, 255),2, 8, 0 );
	
						stringstream ss;
						ss << j<<":"<< contourArea(contours1[j]);
						putText(input, ss.str(), mc, 1, 1, Scalar(255,255,255), 2);
						lesionAreas.push_back(contourArea(contours1[j]));
						mcs.push_back(mc);
						boundRects.push_back(boundRect);
						radii.push_back(radius);
						centers.push_back(center);
						numbers.push_back(j);
					}
					
				}
			}
			lesion.push_back(lesionAreas); 
			lesionAreas.clear();

			mp.push_back(mcs);
			mu.clear();
			mcs.clear();

			allboundRects.push_back(boundRects);
			boundRects.clear();

			allRadii.push_back(radii);
			radii.clear();

			allCenters.push_back(centers);
			centers.clear();

			count.push_back(numbers);
			numbers.clear();
		}				

		else
		{
			Mat contrast=normalization(input);	
			Mat dil=findLesions2(contrast, input);

			//find contours of interest
			vector<vector<Point> > contours;
			findContours(dil, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			Rect boundRect;
			float radius;
			Point2f center;

			vector<Moments> mu(contours.size() );
			Point mc;	

			//for each contour
			for ( size_t j = 0; j < contours.size(); j++)
			{
				mu[j] = moments( contours[j], false ); 
				mc = Point( mu[j].m10/mu[j].m00 , mu[j].m01/mu[j].m00 );

				//for contours with too small area
				if( contourArea(contours[j]) < 300 || (mc.x-10)<0 || (mc.y-10)<0 || (mc.x+10)>input.cols || (mc.y+10)>input.rows)
				{
					//contours.erase(contours.begin() +j); 
					//mc.erase(mc.begin() +j);
					//j=j-1;
				}
				else
				{	
					minEnclosingCircle( contours[j], center, radius);

					if(radius<250)
					{
						drawContours(input, contours, j, Scalar(0, 0, 255),1, 8);
									
						boundRect = boundingRect( Mat(contours[j]) );
						//rectangle(input, boundRect.tl(), boundRect.br(), Scalar(0, 0, 255),2,8,0); 
											
						circle( input, center, radius, Scalar(0, 0, 255), 2, 8, 0 );

						stringstream ss;
						ss << j<<":"<< contourArea(contours[j]);
						putText(input, ss.str(), mc, 1, 1, Scalar(255,255,255), 2);
						lesionAreas.push_back(contourArea(contours[j]));
						mcs.push_back(mc);
						boundRects.push_back(boundRect);
						radii.push_back(radius);
						centers.push_back(center);
						numbers.push_back(j);
					}
				}
			}
			lesion.push_back(lesionAreas); 
			lesionAreas.clear();

			mp.push_back(mcs);
			mu.clear();
			mcs.clear();

			allboundRects.push_back(boundRects);
			boundRects.clear();

			allRadii.push_back(radii);
			radii.clear();

			allCenters.push_back(centers);
			centers.clear();

			count.push_back(numbers);
			numbers.clear();		
		}
	
		//imwrite("results/contours/"+filename, input);			       
    	}

	int true_lesion;
	vector<int> true_lesions;
	vector<int> true_counts;
	vector<Point> true_mp;
	vector<int> true_radii;
	vector<Point2f> true_centers;
	vector<int> growth;
	vector<vector<int>> all_growth;
	//vector<vector<int> all_true_lesions
	
	for (int i = 0; i < lesion.size(); i++)
	{
		if(i==0)
		{	
			for (int a = 0; a < lesion[i].size(); a++)
			{
				true_counts.push_back(count[i][a]);
				true_lesions.push_back(lesion[i][a]);
				true_mp.push_back(mp[i][a]);
				true_radii.push_back(allRadii[i][a]);
				true_centers.push_back(allCenters[i][a]);
			}
		}				
		
		else
		{

			for (int j = 0; j < lesion[i].size(); j++) //compare all candidates with each other (intersecting circle area)
			{		
				double biggest_area=0;int area_to_increase;int value;int del;

				for (int k = 0; k< lesion[i].size(); k++) //calculate 
				{	
					if(count[i][j]!=count[i][k])
					{
						double intersecting_area=area_intersecting_circles(allCenters[i][j], allCenters[i][k], allRadii[i][j], allRadii[i][k]);
						
						if(intersecting_area>0 && intersecting_area>biggest_area)
						{ 
							biggest_area=intersecting_area;
							if(lesion[i][j]>lesion[i][k]){area_to_increase=lesion[i][j];value=lesion[i][k];del=k;}
							else{area_to_increase=lesion[i][k];value=lesion[i][j];del=j;}
						}
					}			
				}

				if(biggest_area>1.5*value) //if lesion is in another, add area
				{	//cout<<i<<":"<<lesion[i][j]<<":"<<area_to_increase<<":"<<value<<endl;
				
					for (int u = 0; u< lesion[i].size(); u++)
					{	
						if(lesion[i][u]==area_to_increase)
						{lesion[i][u]+=value;}
					}
				
					count[i].erase(count[i].begin() +del);
					lesion[i].erase(lesion[i].begin()+del); 
					mp[i].erase(mp[i].begin() +del);
					allRadii[i].erase(allRadii[i].begin()+del);
					allCenters[i].erase(allCenters[i].begin()+del);
				
					j=j-1; 
					area_to_increase=0;value=0;
				}					
			}

			for (int j = 0; j < lesion[i-1].size(); j++) //calculate all distances between mass points to all given lesions from the last picture
			{	
				double biggest_area=0;
				int smallest_dist=1000;		

				for (int m = 0; m< lesion[i].size(); m++)
				{	
					int dist=norm(mp[i-1][j]-mp[i][m]);
					if(dist<smallest_dist){smallest_dist=dist; true_lesion=m;}				
				}
				
				if(smallest_dist<200) //ignore too far lesions (forget them)
				{
					true_counts.push_back(count[i][true_lesion]);
					true_lesions.push_back(lesion[i][true_lesion]);
					true_mp.push_back(mp[i][true_lesion]);
					true_radii.push_back(allRadii[i][true_lesion]);
					true_centers.push_back(allCenters[i][true_lesion]);
					//intersecting_area.clear();
				}
			}

			

			for (int j = 0; j < count[i].size(); j++) //add to near new lesions to nearest
			{	
				vector<int>::iterator it;
				it=find(true_counts.begin(), true_counts.end(), count[i][j]); //find lesion count[i][j] in true_counts
							
				if(it==true_counts.end()) //if not check distance
				{			
					int smallest_dist=1000;	int freq;			

					for (int m = 0; m< true_counts.size(); m++)
					{	
						freq=std::count(true_counts.begin(), true_counts.end(), true_counts[m]);
						int dist=norm(mp[i][j]-true_mp[m])-allRadii[i][j]-true_radii[m]; //dist between circle borders
						if(dist<smallest_dist){smallest_dist=dist; true_lesion=m;}				
					}

					if(smallest_dist<200) 
					{
						for (int n = 0; n< true_counts.size(); n++)
						{	
							if(true_counts[n]==true_counts[true_lesion]){true_lesions[n]+=lesion[i][j]/freq;}
						}
						count[i].erase(count[i].begin() +j);
						lesion[i].erase(lesion[i].begin()+j); 
						mp[i].erase(mp[i].begin() +j);
						allRadii[i].erase(allRadii[i].begin()+j);
						allCenters[i].erase(allCenters[i].begin()+j);
						j=j-1;
					}		
					
				}
			}

			vector<int> multiple;
			for (int j = 0; j < true_counts.size(); j++) //divide same candidates
			{	
				int freq=std::count(true_counts.begin(), true_counts.end(), true_counts[j]);

				if(freq>1)
				{
					multiple.push_back(true_counts[j]);
					
				}
			}
	
			sort( multiple.begin(), multiple.end() );
			multiple.erase( unique( multiple.begin(), multiple.end() ), multiple.end() );
			
			for(int f=0; f<multiple.size(); f++)
			{
				int sum_lesion=0;

				for(int k=0; k<true_counts.size(); k++)
				{
					if(multiple[f]==true_counts[k])
					{
						sum_lesion+=lesion[i-1][k];
					}
				}

				for(int g=0; g<true_counts.size(); g++)
				{
					if(multiple[f]==true_counts[g])
					{	
						true_lesions[g]=(double(lesion[i-1][g])/double(sum_lesion)*true_lesions[g]);
					}
				}
				sum_lesion=0;
			}
			multiple.clear();

			if(total_brightness[i][0]>=100) //daylight images
			{ 
				for (int j = 0; j < count[i].size(); j++) //find new candidates or delete them
				{	
					vector<int>::iterator it;
					it=find(true_counts.begin(), true_counts.end(), count[i][j]); //find lesion count[i][j] in true_counts
							
					if(it==true_counts.end()) //if not check area
					{
						if(lesion[i][j]<500) //delete too small new lesions
						{
							count[i].erase(count[i].begin() +j);						
							lesion[i].erase(lesion[i].begin()+j); 
							mp[i].erase(mp[i].begin() +j);
							allRadii[i].erase(allRadii[i].begin()+j);
							allCenters[i].erase(allCenters[i].begin()+j);

							j=j-1;
						}

						else //new lesion with sufficient area that far away from existing
						{
							true_counts.push_back(count[i][j]);
							true_lesions.push_back(lesion[i][j]);
							true_mp.push_back(mp[i][j]);
							true_radii.push_back(allRadii[i][j]);
							true_centers.push_back(allCenters[i][j]);		
						}
					}
				}	
			}
			
			else //night images
			{	
				for (int j = 0; j < count[i-1].size(); j++) //delete new lesions and calculate growth of proofed lesions
				{				
				
						if(total_brightness[i-1][0]>=100) //precursor image is bright
						{growth.push_back(lesion[i-1][j]);}
						else{growth.push_back(true_lesions[j]-lesion[i-1][j]);} //save growth
				
				}
			}					
		}

		if(growth.size()==0)
		{	
			for (int u=0; u<true_counts.size(); u++)
			{
				growth.push_back(0);
			}
			all_growth.push_back(growth);growth.clear();
		}
		else{all_growth.push_back(growth);growth.clear();}	
		

		for (int j = 0; j < count[i].size(); j++)
		{
			count[i].erase(count[i].begin()+j);
			lesion[i].erase(lesion[i].begin()+j);
			mp[i].erase(mp[i].begin()+j);
			allRadii[i].erase(allRadii[i].begin()+j);
			allCenters[i].erase(allCenters[i].begin()+j);
			j=j-1;
		}

		count[i].insert(count[i].begin(),true_counts.begin(), true_counts.end());	
		lesion[i].insert(lesion[i].begin(),true_lesions.begin(), true_lesions.end());
		mp[i].insert(mp[i].begin(),true_mp.begin(), true_mp.end());
		allRadii[i].insert(allRadii[i].begin(),true_radii.begin(), true_radii.end());
		allCenters[i].insert(allCenters[i].begin(),true_centers.begin(), true_centers.end());
		
		true_counts.clear();
		true_lesions.clear();
		true_mp.clear();
		true_radii.clear();
		true_centers.clear();
			
	}
	
	for (int j = 0; j < all_growth.size(); j++) 
	{
		for (int k = 0; k < all_growth[j].size(); k++) 
		{	
			if(total_brightness[j][0]<=100){
			if(total_brightness[j-1][0]>=100){lesion[j][k]=all_growth[j][k];} 
			else{lesion[j][k]=all_growth[j][k]+lesion[j-1][k];}}
		}
	}	

	cout<<endl;

	for (int i = 0; i < lesion.size(); i++)
	{	
		if(total_brightness[i][0]<=100){cout<<"\033[1;31m"<<timepoints[i]<<"\t"<<taf[i]<<"\033[0m"<<"\t";}
		else{cout<<timepoints[i]<<"\t"<<taf[i]<<"\t";}

		if(i==0)
		{	
			for (int a = 0; a < lesion[i].size(); a++)
			{
				cout<<lesion[i][a]<<"\t";		
			}
		}				
		
		else
		{	
			for (int j = 0; j < lesion[i].size(); j++)
			{
				cout<<lesion[i][j]<<"\t";

			}
		}
		cout<<endl;
	} 

	for (int i = 0; i < lesion.size(); i++)
	{
		if(i==(lesion.size()-1))
		{	
			for (int j = 0; j < count[i].size(); j++)
			{	

				cout<<count[i][j]<<",";
			}
			cout<<endl;

			for (int j = 0; j < lesion[i].size(); j++)
			{	

				cout<<lesion[i][j]<<",";
			}
			cout<<endl;

			for (int j = 0; j < mp[i].size(); j++)
			{	

				cout<<"Point("<<mp[i][j]<<"),";
			}
			cout<<endl;
	
			for (int j = 0; j < allRadii[i].size(); j++)
			{	

				cout<<allRadii[i][j]<<",";
			}
			cout<<endl;
		
			for (int j = 0; j < allCenters[i].size(); j++)
			{	

				cout<<"Point2f("<<allCenters[i][j]<<"),";
			}
			cout<<endl;
		}
	}			
		
		
	for ( size_t m = 0; m < lesion.size(); m++)
	{
		Mat input = imread(filenames[m]);

		if(!input.data)
	    	cerr << "Problem loading image!!!" << endl;

		last = filenames[m].find_last_of('/');
	  	string filename = filenames[m].substr(last+1, string::npos);

		if(total_brightness[m][0]<=100)
		{
			Mat contrast=normalization(input);
			Mat dil=findLesions1(contrast, input);

			vector<vector<Point> > contours1;
			findContours(dil, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			Rect boundRect;
			float radius;
			Point2f center;

			Point a;
			vector<int>multiple;
		
			for (int o = 0; o < lesion[m].size(); o++)
			{	
				stringstream ss;
				ss << o<<":"<< lesion[m][o];				

				drawContours(input, contours1, count[m][o], cv::Scalar(0, 0, 255),1, 8);
				
				boundRect = boundingRect( Mat(contours1[count[m][o]]) );
				//rectangle(input, boundRect.tl(), boundRect.br(), Scalar(0, 0, 255),2,8,0); 
				minEnclosingCircle( contours1[count[m][o]], center, radius);					
				circle( input, center, radius, Scalar(0, 0, 255), 2, 8, 0 );

				int freq=std::count(count[m].begin(), count[m].end(), count[m][o]);
				if(freq>1)
				{	
					multiple.push_back(count[m][o]);	
				}
				
				else
				{
					putText(input, ss.str(), (mp[m][o]), 1, 1, Scalar(255,255,255), 2);
				}
				freq=0;
				
			}

			sort( multiple.begin(), multiple.end() );
			multiple.erase( unique( multiple.begin(), multiple.end() ), multiple.end() );
			
			for(int f=0; f<multiple.size(); f++)
			{
				for(int k=0; k<lesion[m].size(); k++)
				{
					stringstream ss;
					ss <<k<<":"<< lesion[m][k];

					if(multiple[f]==count[m][k])
					{
						putText(input, ss.str(), (mp[m][k]+a), 1, 1,Scalar(255,255,255), 2);
						a+=Point(0,15);
					}
				}				
				a=Point(0,0);
			}
		}
	
		else
		{	
			Mat contrast=normalization(input);
			Mat dil=findLesions2(contrast, input);

			vector<vector<Point> > contours;
			findContours(dil, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);	

			Rect boundRect;
			float radius;
			Point2f center;

			Point a;
			vector<int>multiple;
			
		
			for (int o = 0; o < lesion[m].size(); o++)
			{	
				stringstream ss;
				ss << o<<":"<< lesion[m][o];				

				drawContours(input, contours, count[m][o], cv::Scalar(0, 0, 255),1, 8);

				boundRect = boundingRect( Mat(contours[count[m][o]]) );
				//rectangle(input, boundRect.tl(), boundRect.br(), Scalar(0, 0, 255),2,8,0); 
				minEnclosingCircle( contours[count[m][o]], center, radius);					
				circle( input, center, radius, Scalar(0, 0, 255), 2, 8, 0 );

				int freq=std::count(count[m].begin(), count[m].end(), count[m][o]);
				if(freq>1)
				{	
					multiple.push_back(count[m][o]);	
				}
				
				else
				{
					putText(input, ss.str(), (mp[m][o]), 1, 1, Scalar(255,255,255), 2);
				}
				freq=0;
				
			}

			sort( multiple.begin(), multiple.end() );
			multiple.erase( unique( multiple.begin(), multiple.end() ), multiple.end() );
			
			for(int f=0; f<multiple.size(); f++)
			{
				for(int k=0; k<lesion[m].size(); k++)
				{
					stringstream ss;
					ss <<k<<":"<< lesion[m][k];

					if(multiple[f]==count[m][k])
					{
						putText(input, ss.str(), (mp[m][k]+a), 1, 1,Scalar(255,255,255), 2);
						a+=Point(0,15);	
					}
				}				
				a=Point(0,0);
			}
			

		}
		imwrite("/home/jpfeil/Leaves/results/"+experiment+"/"+filename, input);
		//imwrite("results/filtered_lesions/"+filename, input);
	}
	

	


	/*string yamlName = "results.yaml";
	//yaml file
	FileStorage fs("results/" + yamlName, FileStorage::WRITE);
	fs << "Experiment" << experiment;
	fs << "Exact Time" << timepoints;
	
	//fs << "Lesion"<<;
	for (int i = 0; i < lesion.size(); i++)
	{
		fs << "Time after Inoculation in minutes"<<taf[i]; 
		fs <<"Lesion areas "<<lesion[i];	
	}
	fs.release(); // explicit close	*/
	
		
}






