#include <QApplication>
#include "wObject.h"
#include "Display/MainWindow.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	//read data
	/*std::vector<std::string> filename;
	filename.push_back("original_data/000000.ply");
	//filename.push_back("000000.ply");
    filename.push_back("original_data/000003.ply");
	filename.push_back("original_data/000001.ply");
	filename.push_back("original_data/000002.ply");
	filename.push_back("original_data/000004.ply");
	filename.push_back("original_data/000005.ply");

	int num = 6;
	wObject wo[6];
	std::vector<int> sum_vertex_num(num + 1);
	int tot_size = 0;
	sum_vertex_num[0] = 0;
	for (int i = 0; i < num; i++)
	{
		wo[i].ReadMeshPly(filename[i]);
		tot_size += wo[i].get_vertex_size();
		sum_vertex_num[i + 1] = tot_size;
	}

	int p = 0;
	std::vector<VBOType> vbo(tot_size);
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < wo[i].get_vertex_size(); j++)
		{
			vbo[p].posConf.x = wo[i].get_vertex(j)[0];
			vbo[p].posConf.y = wo[i].get_vertex(j)[1];
			vbo[p].posConf.z = wo[i].get_vertex(j)[2];
			memcpy(&vbo[p].colorTime.x, &wo[i].get_color(j), sizeof(cv::Vec4b));
			vbo[p].colorTime.y = i;
			//memcpy(&rgba, &vbo[p].colorTime.x, sizeof(cv::Vec4b));
			vbo[p].normalRad.x = wo[i].get_normal(j)[0];
			vbo[p].normalRad.y = wo[i].get_normal(j)[1];
			vbo[p].normalRad.z = wo[i].get_normal(j)[2];
			p++;
		}
	}*/


	MainWindow w;
	//w.CreateRegisterDeformation(vbo, sum_vertex_num);
	w.show();



	return a.exec();
}
