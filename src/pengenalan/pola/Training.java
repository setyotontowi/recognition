/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pengenalan.pola;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import javax.swing.JFileChooser;
import javax.swing.table.DefaultTableModel;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_PCA_DATA_AS_ROW;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.PCA;
import org.bytedeco.javacpp.opencv_core.Point2d;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.core.Core;
import static org.opencv.core.Core.PCACompute;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.objdetect.HOGDescriptor;


/**
 *
 * @author Tontowi
 */
public class Training extends javax.swing.JFrame {

    /**
     * Creates new form Training
     * @params
     * classifier : every folder in the directory
     * target : every target for every folder
     * temptarget : all target listed (not for every folder)
     */
    
    JFileChooser file;
    String dir;
    ArrayList<String> classifier = new ArrayList();
    ArrayList<ArrayList<Integer>> target = new ArrayList<>();
    public static ArrayList<ArrayList<Integer>> temptarget = new ArrayList<>();
    public static ArrayList<String> namatarget = new ArrayList<>();
    
    
//    Mat target;
    int y;
    int big=0;
    
    Mat descriptors;
    Mat targetfinal;
    public static ANN_MLP ann;
    public static int cs;
    public static int bs;
    public static int ws;
    public static int bst;
    public static int bin;
    public static int chooser;
    public static int prepo = 0;
    
    public static String[][] pola;
    
    public Training(){
        initComponents();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
      public BufferedImage matToBufferedImage(Mat frame){
            frame.convertTo(frame, CvType.CV_8UC1);
            int type = 0;
            if(frame.channels() == 1){
                type = BufferedImage.TYPE_BYTE_GRAY;
//                System.out.println("gray");
            }
            else if(frame.channels() == 3){
                type = BufferedImage.TYPE_3BYTE_BGR;
//                System.out.println("bgr");
            }
            else{
//                System.out.println("not both");
            }
            //System.out.println(type);
            BufferedImage image = new BufferedImage(frame.width(), frame.height(), type);
            //WritableRaster raster= image.getRaster();
            //DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
            byte[] data = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
            frame.get(0, 0, data);
            return image;
      }
    
      public opencv_core.Mat bufferedImageToMat(BufferedImage bi){
          OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
          return cv.convertToMat(new Java2DFrameConverter().convert(bi));
      }
    
      public int sumFile(String fold){
          int temp=0;
          File folder = new File(fold);
          for(File entry : folder.listFiles()){
              temp++;
          }
          return temp;
      }
    
      public String[] seeFile(String fold, int totaldata){
          String[] dirtemp = new String[totaldata];
          File folder = new File(fold);
          int j=0;
          for(File entry:folder.listFiles()){
                dirtemp[j] = String.valueOf(folder+"\\"+entry.getName());
                j++;
          }
          
          return dirtemp;
      }
      
      public Mat assignTarget(ArrayList<ArrayList<Integer>> t){
          //t merupakan keseluruhan array, array level pertama menunjukkan banyaknya data
          //array level kedua menunjukkan banyaknya digit target
          //assigntarget adalah menaruh semua target ke dalam MAT
          //original Mat assing new mat tsize t0size cvtype
          
          Mat assign = new Mat(t.size(), t.get(0).size(), CvType.CV_32FC1);
          for(int i=0; i<t.size(); i++){
              for(int j=0; j<t.get(0).size(); j++){
                  assign.put(i, j, t.get(i).get(j));
              }
          }
          return assign;
//          System.out.println(t.get(0).get(0));
//          System.out.println(t.size());
          
      }
      
      public String intToBiner(int x){
        ArrayList<Integer> b = new ArrayList();
        if(x==0){
            b.add(0);
        }
        int in=0;
        while(x>0){
            y = x%2;
            b.add(y);
            x = (int) Math.ceil(x/2);
            in++;
        }
        if(in>big){
            big=in;
        }
        while(b.size()<big){
            b.add(0);
        }
        
        Collections.reverse(b); //hasil binerisasi dibaca dari belakang (reverse)
        temptarget.add(b);
        return Arrays.toString(b.toArray());
    }
      
    public Mat computeHOGPCA(String url, ArrayList output, int idk){
        Mat img = Imgcodecs.imread(url);
        Mat deschogpca;
        Size size = new Size(64,64);
        Imgproc.resize(img, img, size);
        
        //input parameter from ekstraksi ciri tab
        //HISTOGRAM OF ORIENTED GRADIENT
        cs  = Integer.parseInt(cellsize.getText());
        bs  = Integer.parseInt(blocksize.getText());
        ws  = Integer.parseInt(winsize.getText());
        bst = Integer.parseInt(blockstride.getText());
        bin  = Integer.parseInt(bininput.getText());
        
        Size window = new Size(ws, ws);
        Size cell   = new Size(cs, cs);
        Size block  = new Size(bs, bs);
        Size stride = new Size(bst, bst);
        
        HOGDescriptor hog = new HOGDescriptor(window, block, stride, cell, bin);
        MatOfFloat descriptor = new MatOfFloat();
        hog.compute(img, descriptor);
        
        deschogpca = descriptor.t();
        
        //PRINCIPAL COMPONENT ANALYSIS
        BufferedImage convj = matToBufferedImage(deschogpca);
        opencv_core.Mat fins =  new opencv_core.Mat(deschogpca.rows(), deschogpca.cols(), CvType.CV_8UC1 );
        fins = bufferedImageToMat(convj);
        Mat descfinal = principalComponentAnalysis(fins, idk);
        
        if(idk==1){
            target.add(output);
        }
        
        return descfinal;
    }
    
    public Mat computeHOG(String url, ArrayList output, int idk, int cs, int bs, int ws, int bst, int bin){
        //idk is classifier for target and training
        
        //read image and resize
        Mat img = Imgcodecs.imread(url);
        Mat deschog;
        Size size = new Size(64,64);
        Imgproc.resize(img, img, size);
        
       
        Imgproc.medianBlur(img, img, 3);
        Imgproc.cvtColor(img, img, opencv_imgproc.COLOR_BGR2GRAY);;
        Imgproc.equalizeHist(img, img);
        Imgproc.threshold(img, img, 160, 255, opencv_imgproc.THRESH_BINARY);
        Imgproc.Canny(img, img, 10, 100, 3, true);
        
        
        //input parameter from ekstraksi ciri tab
        Size window = new Size(ws, ws);
        Size cell   = new Size(cs, cs);
        Size block  = new Size(bs, bs);
        Size stride = new Size(bst, bst);
        
        //Proses HOG
        HOGDescriptor hog = new HOGDescriptor(window, block, stride, cell, bin);
        MatOfFloat descriptor = new MatOfFloat();
        hog.compute(img, descriptor);
        
        deschog = descriptor.t();
        
        //Assign to target
        if(idk==1){
            target.add(output);
        }
        
        return deschog;
    }
    
    public Mat computePCA(String url, ArrayList output, int idk){
        //Read Image
        opencv_core.Mat imgb = opencv_imgcodecs.imread(url);
        
        UByteIndexer colorIndexer = imgb.createIndexer();
        for(int y=0; y<imgb.rows(); y++){
            for(int x=0; x<imgb.cols();x++){
                double color = colorIndexer.get(y,x);
            }
        }
        
        Mat img = Imgcodecs.imread(url);
        Size sizes = new Size(64,64);
        Imgproc.resize(img, img, sizes);
        
       
        Imgproc.medianBlur(img, img, 3);
        Imgproc.cvtColor(img, img, opencv_imgproc.COLOR_BGR2GRAY);;
        Imgproc.equalizeHist(img, img);
        Imgproc.threshold(img, img, 160, 255, opencv_imgproc.THRESH_BINARY);
        Imgproc.Canny(img, img, 10, 100, 3, true);
        
        opencv_core.Size size = new opencv_core.Size(64,64);
        opencv_imgproc.resize(imgb, imgb, size);
        
        // preprocessing
        opencv_imgproc.medianBlur(imgb, imgb, 3);
        opencv_imgproc.cvtColor(imgb, imgb, opencv_imgproc.COLOR_BGR2GRAY);
        opencv_imgproc.equalizeHist(imgb, imgb);
        opencv_imgproc.threshold(imgb, imgb, 160, 255, opencv_imgproc.THRESH_BINARY);
        opencv_imgproc.Canny(imgb, imgb, 10, 100, 3, true);
        
//        HasilGambar hsl = new HasilGambar();
//        hsl.showgambar(img);
//        hsl.setVisible(true);
        
        //perhitungan menggunakan method PCA
        Mat descpca = principalComponentAnalysis(imgb, idk);
        
        if(idk==1){
            target.add(output);
        }
        
        return descpca;
        
    }
    
    public Mat principalComponentAnalysis(opencv_core.Mat input, int idk){
        PCA pca_analysis;
        opencv_core.Mat mean;
        opencv_core.Mat eigenVector;
        opencv_core.Mat eigenValue;
        
        //construct a buffer
        opencv_core.Mat data_pts = new opencv_core.Mat(input.rows(), 2, CvType.CV_64FC1);
        opencv_core.Mat placeholder = new opencv_core.Mat();
        
        UByteIndexer dataIndexer = input.createIndexer();
        DoubleIndexer data = data_pts.createIndexer();
        for(int i=0; i<2; i++){
            data.put(i,0, dataIndexer.get(i,0));
            data.put(i,1, dataIndexer.get(i,1));
        }
        dataIndexer.release();
        data.release();
        
        //perform PCA
        ArrayList<Point2d> eigen_vecs = new ArrayList(2);
        ArrayList<Float> eigen_val = new ArrayList(2);
        
        pca_analysis    = new PCA(input, placeholder, CV_PCA_DATA_AS_ROW);
        mean            = pca_analysis.mean();
        eigenVector     = pca_analysis.eigenvectors();
        eigenValue      = pca_analysis.eigenvalues();
        
        FloatIndexer mean_idx = mean.createIndexer();
        FloatIndexer eigenVectorIndexer = eigenVector.createIndexer();
        FloatIndexer eigenValueIndexer = eigenValue.createIndexer();
        
        for(int i=0; i<2; i++){
            eigen_vecs.add(new Point2d(eigenVectorIndexer.get(i, 0), eigenVectorIndexer.get(i,1)));
            eigen_val.add(eigenValueIndexer.get(0,i));
        }
        
        double x1 = eigen_vecs.get(0).x();
        double y1 = eigen_vecs.get(0).y();
        double x2 = eigen_vecs.get(1).x();
        double y2 = eigen_vecs.get(1).y();
        
        eigenVectorIndexer.release();
        eigenValueIndexer.release();
        
        Mat result = new Mat(1, 4,CvType.CV_32FC1);

        result.put(0,0,x1);
        result.put(0,1,y1);
        result.put(0,2,x2);
        result.put(0,3,y2);
        
        System.out.println(result.dump());
        
        return result;
    }
    
    public void trainANN(Mat input, Mat target, int targetlength, int hiddenlayer){
        Mat layers = new Mat(3,1, CvType.CV_32FC1);
        //yang jadi inputannya di form adalah ini
        int [] layers_ann = new int[]{input.cols(), hiddenlayer, targetlength};
        for(int i=0; i<3; i++){layers.put(i,0,layers_ann[i]);}
        
        //System.out.println("success #1 \n");
        
        ann = ANN_MLP.create();
        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1.0, 1.0);
        TermCriteria tc = new TermCriteria(TermCriteria.EPS, 1000, 0.00001);
        ann.setTermCriteria(tc);
       // System.out.println(input.dump()+" ----- "+target.dump());
        ann.train(input, Ml.ROW_SAMPLE, target);
        
        log.append("training success \n");
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        buttonGroup1 = new javax.swing.ButtonGroup();
        jTabbedPane1 = new javax.swing.JTabbedPane();
        jPanel1 = new javax.swing.JPanel();
        inputdataset = new javax.swing.JButton();
        srclabel = new javax.swing.JTextField();
        jScrollPane1 = new javax.swing.JScrollPane();
        tablecontent = new javax.swing.JTable();
        reset = new javax.swing.JButton();
        jPanel2 = new javax.swing.JPanel();
        jPanel3 = new javax.swing.JPanel();
        HOGcheck = new javax.swing.JCheckBox();
        PCAcheck = new javax.swing.JCheckBox();
        jPanel4 = new javax.swing.JPanel();
        jLabel3 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        jLabel5 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        jLabel7 = new javax.swing.JLabel();
        bininput = new javax.swing.JTextField();
        blockstride = new javax.swing.JTextField();
        blocksize = new javax.swing.JTextField();
        winsize = new javax.swing.JTextField();
        cellsize = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        jLabel9 = new javax.swing.JLabel();
        hiddenlayer = new javax.swing.JTextField();
        log = new javax.swing.JTextArea();
        train = new javax.swing.JButton();
        jButton4 = new javax.swing.JButton();
        jButton1 = new javax.swing.JButton();
        jLabel8 = new javax.swing.JLabel();
        prep = new javax.swing.JComboBox<>();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jTabbedPane1.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        inputdataset.setText("Input Dataset");
        inputdataset.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                inputdatasetActionPerformed(evt);
            }
        });

        srclabel.setEditable(false);
        srclabel.setText("src");

        tablecontent.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {

            },
            new String [] {
                "Data", "Classifier", "Classifier Name"
            }
        ) {
            Class[] types = new Class [] {
                java.lang.String.class, java.lang.String.class, java.lang.String.class
            };

            public Class getColumnClass(int columnIndex) {
                return types [columnIndex];
            }
        });
        jScrollPane1.setViewportView(tablecontent);
        if (tablecontent.getColumnModel().getColumnCount() > 0) {
            tablecontent.getColumnModel().getColumn(0).setHeaderValue("Data");
            tablecontent.getColumnModel().getColumn(1).setHeaderValue("Classifier");
            tablecontent.getColumnModel().getColumn(2).setHeaderValue("Classifier Name");
        }

        reset.setText("Reset");
        reset.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                resetActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(inputdataset)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(srclabel))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(reset)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(inputdataset)
                    .addComponent(srclabel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 336, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(reset)
                .addContainerGap())
        );

        jTabbedPane1.addTab("Data Input", jPanel1);

        jPanel3.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        HOGcheck.setText("Histogram Of Oriented Gradient");
        HOGcheck.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                HOGcheckStateChanged(evt);
            }
        });
        HOGcheck.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                HOGcheckMouseClicked(evt);
            }
        });
        HOGcheck.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                HOGcheckActionPerformed(evt);
            }
        });

        PCAcheck.setText("Principal Component Analysis");

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(HOGcheck)
                    .addComponent(PCAcheck))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(HOGcheck)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(PCAcheck)
                .addContainerGap())
        );

        jPanel4.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        jLabel3.setText("Cell Size");

        jLabel4.setText("Window Size");

        jLabel5.setText("Block Size");

        jLabel6.setText("Block Stride");

        jLabel7.setText("Bin");

        bininput.setEnabled(false);

        blockstride.setEnabled(false);

        blocksize.setEnabled(false);

        winsize.setEnabled(false);

        cellsize.setEnabled(false);

        jLabel2.setText("HOG Properties");

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel2)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel4Layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(jLabel5, javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel3, javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel7, javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel6, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 91, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel4, javax.swing.GroupLayout.Alignment.LEADING))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(blockstride, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(winsize, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(blocksize, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(bininput, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cellsize, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel2)
                .addGap(16, 16, 16)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cellsize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel3))
                .addGap(5, 5, 5)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(blocksize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel5, javax.swing.GroupLayout.PREFERRED_SIZE, 9, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(bininput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel7))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(winsize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel4))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(blockstride, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel6))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        jLabel1.setFont(new java.awt.Font("Source Sans Pro", 1, 12)); // NOI18N
        jLabel1.setText("Ekstraksi Fitur");

        jLabel9.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        jLabel9.setText("Hidden Layer");

        hiddenlayer.setText("100");

        log.setEditable(false);
        log.setColumns(20);
        log.setRows(5);
        log.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        train.setText("Start Training");
        train.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                trainActionPerformed(evt);
            }
        });

        jButton4.setText("Testing");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });

        jButton1.setText("Save");

        jLabel8.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        jLabel8.setText("Preprocessing");

        prep.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Median Blur, Grayscale, Equalize Hist, Canny", "Grayscale, Equalize Hist, Canny", "Basic" }));

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(log)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(jPanel3, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(jPanel2Layout.createSequentialGroup()
                                .addComponent(train)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(jButton4)
                                .addGap(71, 71, 71)
                                .addComponent(jButton1, javax.swing.GroupLayout.DEFAULT_SIZE, 65, Short.MAX_VALUE))
                            .addComponent(prep, javax.swing.GroupLayout.Alignment.LEADING, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addGap(10, 10, 10))
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(jPanel2Layout.createSequentialGroup()
                                .addComponent(jLabel9)
                                .addGap(18, 18, 18)
                                .addComponent(hiddenlayer, javax.swing.GroupLayout.PREFERRED_SIZE, 95, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(jLabel1)
                            .addComponent(jLabel8))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addComponent(jPanel4, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addComponent(jLabel8)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(prep)
                        .addGap(33, 33, 33)
                        .addComponent(jLabel1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jPanel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGap(18, 18, 18)
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jLabel9)
                            .addComponent(hiddenlayer, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(18, 18, 18)
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(train, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jButton4, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 28, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addComponent(jPanel4, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(log, javax.swing.GroupLayout.DEFAULT_SIZE, 128, Short.MAX_VALUE)
                .addContainerGap())
        );

        jTabbedPane1.addTab("Training Properties", jPanel2);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jTabbedPane1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jTabbedPane1))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void HOGcheckActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_HOGcheckActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_HOGcheckActionPerformed

    private void HOGcheckMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_HOGcheckMouseClicked
        // TODO add your handling code here:
        if(HOGcheck.getSelectedObjects()!= null){
            cellsize.setEnabled(true);
            winsize.setEnabled(true);
            blocksize.setEnabled(true);
            blockstride.setEnabled(true);
            bininput.setEnabled(true);

            cellsize.setText("16");
            winsize.setText("64");
            blocksize.setText("32");
            blockstride.setText("16");
            bininput.setText("9");
        }
        else{
            cellsize.setEnabled(false);
            winsize.setEnabled(false);
            blocksize.setEnabled(false);
            blockstride.setEnabled(false);
            bininput.setEnabled(false);
        }
    }//GEN-LAST:event_HOGcheckMouseClicked

    private void HOGcheckStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_HOGcheckStateChanged

    }//GEN-LAST:event_HOGcheckStateChanged

    private void resetActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_resetActionPerformed
        // TODO add your handling code here:
        DefaultTableModel model = (DefaultTableModel) tablecontent.getModel();
        if(model.getRowCount()!= 0){
            for(int i=model.getRowCount()-1; i>=0; i--){
                model.removeRow(i);
            }
            classifier.clear();
            temptarget.clear();
        }

    }//GEN-LAST:event_resetActionPerformed

    private void inputdatasetActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_inputdatasetActionPerformed
        // Add Folder Directory
        file = new JFileChooser();
        file.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        file.showOpenDialog(this);
        dir = file.getSelectedFile().toString();

        srclabel.setText(dir);

        //List Folder
        File files = new File(dir);
        String[] directories = files.list(new FilenameFilter(){
            public boolean accept(File current, String name){
                return new File(current, name).isDirectory();
            }
        });

        //Generate Binary for Target
        int dirlength = directories.length;
        ArrayList<String> c = new ArrayList();
        for(int i=dirlength-1; i>=0; i--){
            String a = intToBiner(i+1);
            c.add(a);
            //            temptarget2.add(i+1);
            //c.add(a);
        }
        Collections.reverse(c);
        Collections.reverse(temptarget);

        // Add it into Table
        DefaultTableModel model = (DefaultTableModel) tablecontent.getModel();
        pola = new String[2][dirlength];
        for(int i=0; i<dirlength; i++){
            model.addRow(new Object[]{directories[i], c.get(i),"Pola "+directories[i]});
            classifier.add(dir+"\\"+directories[i]);
//            namatarget.add((String) model.getValueAt(i, 2));  
            pola[0][i] = String.valueOf(i);
//            pola[1][i] = (String) model.getValueAt(i, 2);
        }
        
        
        
        //seharusnya target diambil dari table

            
    }//GEN-LAST:event_inputdatasetActionPerformed

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        // TODO add your handling code here:
//        System.out.println(chooser);
        Testing testing= new Testing();
        testing.setVisible(true);
    }//GEN-LAST:event_jButton4ActionPerformed

    private void trainActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_trainActionPerformed
        // TODO add your handling code here:
        long startTime = System.currentTimeMillis();
        descriptors = new Mat();
        targetfinal = new Mat();
        
        int dirlength = classifier.size();
        DefaultTableModel model = (DefaultTableModel) tablecontent.getModel();
        namatarget.clear();
        for(int i=0; i<dirlength; i++){
            namatarget.add((String) model.getValueAt(i, 2));
            pola[1][i] = (String) model.getValueAt(i, 2);
        }

        if(HOGcheck.isSelected() && PCAcheck.isSelected()){
            //Do some HOG then PCA
            log.append("HOG and PCA is Selected \n");

            for(int i=0; i<classifier.size();i++){
                int num = sumFile(classifier.get(i));
                String [] tempdir = seeFile(classifier.get(i), num);
                for(int j=0; j<num; j++){
                    //String d = Arrays.toString(temptarget.get(i).toArray());
                    //                   descriptors.push_back(computeHOGPCA(tempdir[j], temptarget.get(i), 1));
                }
            }

//            log.append("Training data : rows= "+descriptors.rows()+" cols= "+descriptors.cols());
        }
        else if(HOGcheck.isSelected()){
            //Do some HOG
            chooser = 1;
            cs  = Integer.parseInt(cellsize.getText());
            bs  = Integer.parseInt(blocksize.getText());
            ws  = Integer.parseInt(winsize.getText());
            bst = Integer.parseInt(blockstride.getText());
            bin  = Integer.parseInt(bininput.getText());

            log.append("HOG is Selected with parameters : \n"
                + "winsize = "+ ws +"\n"
                + "cellsize = "+ cs+"\n"
                + "blocksize = "+ bs +"\n"
                + "blockstride = "+ bst +"\n"
                + "bin = "+ bin +"\n");

            for(int i=0; i<classifier.size();i++){
                int num = sumFile(classifier.get(i));
                String [] tempdir = seeFile(classifier.get(i), num);
                for(int j=0; j<num; j++){
                    //String d = Arrays.toString(temptarget.get(i).toArray());
                    descriptors.push_back(computeHOG(tempdir[j], temptarget.get(i), 1, cs, bs, ws, bst, bin));
                }
            }

            log.append("Training data : rows= "+descriptors.rows()+" cols= "+descriptors.cols()+"\n");
        }
        else if(PCAcheck.isSelected()){
            //Do some PCA
            chooser = 2;
            log.append("PCA is Selected \n");

            for(int i=0; i<classifier.size();i++){
                int num = sumFile(classifier.get(i));
                String [] tempdir = seeFile(classifier.get(i), num);
                for(int j=0; j<num; j++){
                    //String d = Arrays.toString(temptarget.get(i).toArray());
                    descriptors.push_back(computePCA(tempdir[j], temptarget.get(i), 1));
                }
            }

            log.append("Training data : rows= "+descriptors.rows()+" cols= "+descriptors.cols()+"\n");
        }

        targetfinal = assignTarget(target);
//        System.out.println(targetfinal.rows() + " = " +targetfinal.cols());

        //untuk training bahan yang dibutuhkan adalah descriptor dan targetfinal
        int hl = Integer.parseInt(hiddenlayer.getText());
        trainANN(descriptors, targetfinal, target.get(0).size(),hl);
        //trainANN(descriptors, targetfinal, target.size());
        long endTime = System.currentTimeMillis();

       log.append("Process Time = "+(endTime-startTime)+"ms \n");
    }//GEN-LAST:event_trainActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Training.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Training.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Training.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Training.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run(){
                new Training().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JCheckBox HOGcheck;
    private javax.swing.JCheckBox PCAcheck;
    private javax.swing.JTextField bininput;
    private javax.swing.JTextField blocksize;
    private javax.swing.JTextField blockstride;
    private javax.swing.ButtonGroup buttonGroup1;
    private javax.swing.JTextField cellsize;
    private javax.swing.JTextField hiddenlayer;
    private javax.swing.JButton inputdataset;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton4;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTabbedPane jTabbedPane1;
    private javax.swing.JTextArea log;
    private javax.swing.JComboBox<String> prep;
    private javax.swing.JButton reset;
    private javax.swing.JTextField srclabel;
    private javax.swing.JTable tablecontent;
    private javax.swing.JButton train;
    private javax.swing.JTextField winsize;
    // End of variables declaration//GEN-END:variables
}
