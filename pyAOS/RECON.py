import numpy
import pyfits
import logging
import traceback
import sys
import time

#xrange now just "range" in python3. 
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

class Reconstructor:
    def __init__(   
        self, dms, wfss, totalSlopes,dmNo, dmTypes,dmActs, dmCond, totalActs,
        runWfs, atmos, learnAtmos, learnIters, filePrefix, saveLearn):
                
        self.dms = dms
        self.totalSlopes = totalSlopes
        self.dmCond = dmCond
        self.totalActs = totalActs
        self.dmNo = dmNo
        self.dmActs = dmActs
        self.dmTypes = dmTypes
        self.learnIters = learnIters
        self.filePrefix=filePrefix
        self.saveLearn = saveLearn
        
        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfs
        if learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss
                
        self.controlMatrix = numpy.zeros((self.totalSlopes, self.totalActs))
        self.controlShape = (self.totalSlopes, self.totalActs)
        
        self.Trecon = 0

    def saveCMat(self):
        filename = self.filePrefix+"/cMat.fits"
        
        cMatHDU = pyfits.PrimaryHDU(self.controlMatrix)
        cMatHDU.header["DMNO"] = self.dmNo
        cMatHDU.header["DMACTS"] = "%s"%list(self.dmActs)
        cMatHDU.header["DMTYPE"]  = "%s"%list(self.dmTypes)
        cMatHDU.header["DMCOND"]  = "%s"%list(self.dmCond)
        
        cMatHDU.writeto(filename, clobber=True)
        # FITS.Write(self.controlMatrix,filename,
#                     extraHeader=[   "DMNO    = %s"%self.dmNo,
#                                     "DMACTS  = %s"%list(self.dmActs),
#                                     "DMTYPE  = %s"%list(self.dmTypes),
#                                     "DMCOND  = %s"%list(self.dmCond)
#                                            ])   

    def loadCMat(self):
        
        filename=self.filePrefix+"/cMat.fits"
        
        # cMatFile = FITS.Read(filename)
#         header = cMatFile[0]["parsed"]
        
        cMatHDU = pyfits.open(filename)[0]
        cMatHDU.verify("fix")
        header = cMatHDU.header
        
        
        try:
            dmActs = dmTypes = dmCond = None
            
            
            dmNo = int(header["DMNO"])
            exec("dmActs = numpy.array(%s)"%cMatHDU.header["DMACTS"])
            exec("dmTypes = %s"%header["DMTYPE"])
            exec("dmCond = numpy.array(%s)"%header["DMCOND"])
            
            if (dmCond==self.dmCond).all()==False:
                raise Exception("DM conditioning Parameter changed - will make new control matrix")
            if (dmActs==self.dmActs).all() !=True or dmTypes != self.dmTypes or dmNo != dmNo:
                logging.warning("loaded control matrix may not be compatibile with \
                                the current simulation. Will try anyway....")
                                
            #cMat = cMatFile[1]
            cMat = cMatHDU.data
            
        except KeyError:
            logging.warning("loaded control matrix header has not created by this ao sim. Will load anyway.....")
            #cMat = cMatFile[1]
            cMat = cMatHDU.data
            
        if cMat.shape != self.controlShape:
            logging.warning("designated control matrix does not match the expected shape")
            raise Exception
        else:
            self.controlMatrix = cMat
    
    def saveIMat(self):

        for dm in xrange(self.dmNo):
            filenameIMat = self.filePrefix+"/iMat_dm%d.fits"%dm
            filenameShapes = self.filePrefix+"/dmShapes_dm%d.fits"%dm
            
            pyfits.PrimaryHDU(self.dms[dm].iMat).writeto(filenameIMat,
                                                        clobber=True)
            pyfits.PrimaryHDU(self.dms[dm].iMatShapes).writeto(filenameShapes,
                                                        clobber=True)
            
            
            # FITS.Write(self.dms[dm].iMat,filenameIMat)
            #FITS.Write(self.dms[dm].iMatShapes,filenameShapes)
    
                                              
    def loadIMat(self):
        
        for dm in xrange(self.dmNo):
            filenameIMat = self.filePrefix+"/iMat_dm%d.fits"%dm
            filenameShapes = self.filePrefix+"/dmShapes_dm%d.fits"%dm
            
            #iMat = FITS.Read(filenameIMat)[1]
            #iMatShapes = FITS.Read(filenameShapes)[1]
            
            iMat = pyfits.open(filenameIMat)[0].data
            iMatShapes = pyfits.open(filenameShapes)[0].data
            
            if iMat.shape != (self.dms[dm].acts,2*self.dms[dm].totalSubaps):
                logging.warning("interaction matrix does not match required required size.")
                raise Exception
            if iMatShapes.shape[-1]!=self.dms[dm].pupilSize:
                logging.warning("loaded DM shapes are not same size as current pupil.")
                raise Exception
            else:
                self.dms[dm].iMat = iMat
                self.dms[dm].iMatShapes = iMatShapes
    

    def reconstruct(self,slopes):
        t=time.time()
        dmCommands = self.controlMatrix.T.dot(slopes)
        self.Trecon += time.time()-t
        return dmCommands
        
    def makeIMat(self,callback, progressCallback):
        acts = 0
        for dm in xrange(self.dmNo):
            logging.info("Creating Interaction Matrix on DM %d..."%dm)
            self.dms[dm].makeIMat(callback=callback, 
                                        progressCallback=progressCallback)

        
        
    def makeCMat(self,loadIMat=True,loadCMat=True, callback=None, 
                        progressCallback=None):
        
        
        if loadIMat:
            try:
                self.loadIMat()
                logging.info("Interaction Matrices loaded successfully")
            except:
                traceback.print_exc()
                logging.warning("Load Interaction Matrices failed - will create new one.")
                self.makeIMat(callback=callback,    
                         progressCallback=progressCallback)
                self.saveIMat()
                logging.info("Interaction Matrices Done")
                
        else:
            
            self.makeIMat(callback=callback, progressCallback=progressCallback)
            logging.info("Interaction Matrices Done")
            
        
        if loadCMat:
            try:
                self.loadCMat()
                logging.info("Command Matrix Loaded Successfully")
            except:
                traceback.print_exc()
                logging.warning("Load Command Matrix failed - will create new one")
                
                self.calcCMat(callback, progressCallback)
                self.saveCMat()
                logging.info("Command Matrix Generated!")
        else:
            logging.info("Creating Command Matrix")
            self.calcCMat(callback, progressCallback)
            logging.info("Command Matrix Generated!")
            


class MVM(Reconstructor):
    '''
    Reconstructs by using DM interaction Matrices to create a control 
    Matrix. 
    Treats each DM seperately, so all DMs try to correct all turbulence.
    '''
    

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a 
        control matrix
        '''
        acts = 0
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts
    
        
            
class WooferTweeter(Reconstructor):
    '''
    Reconstructs a 2 DM system, where 1 DM is of low order, high stroke
    and the other has a higher, but low stroke.
    
    Reconstructs dm commands for each DM, then removes the low order 
    component from the high order commands by propagating back to the 
    slopes corresponding to the lower order DM shape, and propagating 
    to the high order DM shape.
    '''
    
    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Creates control Matrix. 
        Assumes that DM 0 (or 1 if TT used) is low order, 
        and DM 1 (or 2 if TT used) is high order.
        '''

        if self.dmNo==1:
            logging.warning("Woofer Tweeter Reconstruction not valid for 1 dm.")
            return None
        acts = 0
        dmCMats = []
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            if dm != self.dmNo-1:
                self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
                acts+=self.dms[dm].acts
            
            dmCMats.append(dmCMat)
            
            
            
            
        #This it the matrix which converts from Low order DM commands
        #to high order DM commands, via slopes
        lowToHighTransform = self.dms[self.dmNo-2].iMat.T.dot( dmCMats[-2].T )
        print(lowToHighTransform.shape)

        highOrderCMat = dmCMats[-1].T.dot( numpy.identity(self.totalSlopes) - \
                            lowToHighTransform )
                            
        self.controlMatrix[:,acts:acts+self.dms[self.dmNo-1].acts] = highOrderCMat.T
 
    
class LearnAndApply(Reconstructor):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis 
    and on-axis WFS is computed from these slopes. 
    
    Assumes that on-axis sensor is WFS 0
    
    Only works for 1 DM at present
    '''
    
    def __init__(self, dms, wfss, totalSlopes,dmNo,
                dmTypes,dmActs, dmCond,totalActs,runWfs, atmos, learnAtmos,
                learnIters, filePrefix, saveLearn):
                
        Reconstructor.__init__(self, dms, wfss, totalSlopes,dmNo,
                dmTypes,dmActs, dmCond,totalActs,runWfs, atmos, learnAtmos, 
                learnIters, filePrefix, saveLearn)
        
        self.controlShape = (   self.totalSlopes-2*wfss[0].activeSubaps,
                                self.totalActs )
                
    
    def learn(self,callback=None, progressCallback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.empty( (self.learnIters,self.totalSlopes) )
        for i in xrange(self.learnIters):
            self.learnIter=i            
            logging.debug("Learn Iteration %i",i)
            
            scrns = self.moveScrns()
            self.learnSlopes[i] = self.runWfs(scrns)
            
            sys.stdout.write("\rLearn Frame: %d    "%i)
            sys.stdout.flush()
            
            if callback!=None:
                callback()
            if progressCallback!=None:
               progressCallback(i,self.learnIters, "Performing Learn") 
            
        if self.saveLearn:
            #FITS.Write(self.learnSlopes,self.filePrefix+"/learn.fits")
            pyfits.PrimaryHDU(self.learnSlopes).writeto(
                            self.filePrefix+"/learn.fits",clobber=True )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        self.controlMatrix = numpy.empty( (2*self.wfss[0].activeSubaps,
                                            self.totalActs) )
        logging.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logging.info("Done. Creating Tomographic Reconstructor...")
        
        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")
        
        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[0].activeSubaps,
                                2*self.wfss[0].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[0].activeSubaps:,
                                2*self.wfss[0].activeSubaps:    ]
        if progressCallback:
            progressCallback(1,1, "Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.inv(Coffoff)
        
        self.tomoRecon = Conoff.dot(iCoffoff)
        logging.info("Done. Creating full reconstructor....")
        
        #Same code as in "MVM" class to create dm-slopes reconstructor.
        acts = 0
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts
        
        if progressCallback:
            progressCallback(1,1, "Creating full reconstructor")
        self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logging.info("Done.")
        
    def reconstruct(self,slopes):
        
        logging.debug("LA Reconstruction - slopes Shape: %s"%slopes[2*self.wfss[0].activeSubaps:].shape)
        logging.debug("LA Reconstruction - Reconstructor Shape: %s,%s"%self.controlMatrix.shape)
        
        dmCommands = self.controlMatrix.T.dot(
                            slopes[2*self.wfss[0].activeSubaps:])
        
        return dmCommands          
    
class LearnAndApply_NGSTT(Reconstructor):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis 
    and on-axis WFS is computed from these slopes. 
    
    Assumes that on-axis sensor is WFS 0
    
    This reconstructor performs L&A on the LGS slopes, but does a simple
    interaction matrix with the last WFS, which is assumed to be a low order
    Tip Tilt correcting NGS 
    
    Only works for 1 DM at present
    
    ***
    Doesn't work!
    ***
    '''
    
    def __init__(self, dms, wfss, totalSlopes,dmNo,
                dmTypes,dmActs, dmCond,totalActs,runWfs,moveScrns, learnIters,
                filePrefix, saveLearn):
                
        Reconstructor.__init__(self, dms, wfss, totalSlopes,dmNo,
                dmTypes,dmActs, dmCond,totalActs,runWfs,moveScrns, learnIters,
                filePrefix, saveLearn)
        
        self.controlShape = (   self.totalSlopes-2*wfss[0].activeSubaps,
                                self.totalActs )
                
    
    def learn(self,callback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.empty( (self.learnIters,self.totalSlopes) )
        for i in xrange(self.learnIters):
            self.learnIter=i            
            logging.debug("Learn Iteration %i",i)
            
            scrns = self.moveScrns()
            self.learnSlopes[i] = self.runWfs(scrns)
            
            sys.stdout.write("\rLearn Frame: %d    "%i)
            sys.stdout.flush()
            
            if callback!=None:
                callback()
            
        if self.saveLearn:
            #FITS.Write(self.learnSlopes,self.filePrefix+"/learn.fits")
            pyfits.PrimaryHDU(self.learnSlopes).writeto(
                                    self.filePrefix+"/learn.fits",clobber=True)

    def calcCMat(self,callback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        self.controlMatrix = numpy.empty( (2*self.wfss[0].activeSubaps,
                                            self.totalActs) )
        logging.info("Performing Learn....")
        self.learn(callback)
        logging.info("Done. Creating Tomographic Reconstructor...")
        
        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[0].activeSubaps,
                                2*self.wfss[0].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[0].activeSubaps:,
                                2*self.wfss[0].activeSubaps:    ]
        
        iCoffoff = numpy.linalg.inv(Coffoff)
        
        self.tomoRecon = Conoff.dot(iCoffoff)
        logging.info("Done. Creating full reconstructor....")
        
        #Same code as in "MVM" class to create dm-slopes reconstructor.
        acts = 0
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

        
        self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logging.info("Done.")
        
    def reconstruct(self,slopes):
        
        logging.debug("LA Reconstruction - slopes Shape: %s"%slopes[2*self.wfss[0].activeSubaps:].shape)
        logging.debug("LA Reconstruction - Reconstructor Shape: %s,%s"%self.controlMatrix.shape)
        
        dmCommands = self.controlMatrix.T.dot(
                            slopes[2*self.wfss[0].activeSubaps:])
        
        return dmCommands          
    
    



#####################################
#Experimental....
#####################################

class ANN(Reconstructor):
    """
    Reconstructs using a neural net
    Assumes on axis slopes are WFS 0

    Net must be set by setting ``sim.recon.net = net`` before loop is run 
    net object must have a ``run`` method, which accepts slopes and returns
    on Axis slopes
    """

    def calcCMat(self, callback=None, progressCallback=None):

        nSlopes = self.wfss[0].activeSubaps*2 

        self.controlShape = (nSlopes, self.totalActs)
        self.controlMatrix = numpy.zeros((nSlopes, self.totalActs))
        acts = 0
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made 
        reconstructor from slopes. Uses Artificial Neural Network.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM 
        """
        t=time.time()
        offSlopes = slopes[self.wfss[0].activeSubaps*2:]
        onSlopes = self.net.run(offSlopes)
        dmCommands = self.controlMatrix.T.dot(onSlopes)

        self.Trecon += time.time()-t
        return dmCommands


class LearnAndApplyLGS(Reconstructor):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis 
    and on-axis WFS is computed from these slopes. 
    
    Assumes that on-axis sensor is WFS 0
    
    Only works for 1 DM at present
    
    This is a specific implementation designed to test LGS tip tilt 
    prediction. It uses the last 10 frames of off-axis slopes to help predict
    the on-axis slopes.
    '''
    
    def learn(self):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''
        FRAMES = 10
        
        onAxisSize = self.wfss[0].activeSubaps*2
        offAxisSize = self.totalSlopes - self.wfss[0].activeSubaps*2
        
        self.onSlopes = numpy.empty( (self.learnIters+FRAMES,onAxisSize) )
        self.offSlopes = numpy.empty( (self.learnIters+FRAMES,
                                                FRAMES*offAxisSize) )
        self.slopesBuffer = numpy.empty( (FRAMES,offAxisSize) )
        
        for i in xrange(self.learnIters+FRAMES):
            self.learnIter = i
            logging.debug("Learn Iteration %i",i)
            scrns = self.moveScrns()
            slopes = self.runWfs(scrns)
            
            self.onSlopes[i] = slopes[:onAxisSize]            
            self.slopesBuffer[0] = slopes[onAxisSize:]
            self.offSlopes[i] = self.slopesBuffer.flatten()
            self.slopesBuffer = numpy.roll(self.slopesBuffer,1,axis=0)

        self.offSlopes = self.offSlopes[FRAMES:]
        self.onSlopes = self.onSlopes[FRAMES:]

    def calcCMat(self):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''
        self.controlMatrix = numpy.empty( (2*self.wfss[0].activeSubaps,
                                            self.totalActs) )
        logging.info("Performing Learn....")
        self.learn()
        logging.info("Done. Creating Tomographic Reconstructor...")
        onAxisSlopes = self.onSlopes
        iOffAxisSlopes = numpy.linalg.pinv(self.offSlopes)
        
        tomoRecon = onAxisSlopes.T.dot(iOffAxisSlopes.T)
        logging.info("Done. Creating full reconstructor....")
        
        
        #Same code as in "MVM" class to create dm-slopes reconstructor.
        acts = 0
        for dm in xrange(self.dmNo):
            dmIMat = self.dms[dm].iMat
            
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmCond[dm])
            
            self.controlMatrix[:,acts:acts+self.dms[dm].acts] = dmCMat
            acts += self.dms[dm].acts

        
        self.controlMatrix = (self.controlMatrix.T.dot(tomoRecon)).T
        logging.info("Done.")
        
    def reconstruct(self,slopes):
        
        logging.debug("LA Reconstruction - slopes Shape: %s"%slopes[2*self.wfss[0].activeSubaps:].shape)
        logging.debug("LA Reconstruction - Reconstructor Shape: %s,%s"%self.controlMatrix.shape)

        self.slopesBuffer[0] = slopes[self.wfss[0].activeSubaps*2:]
        
        dmCommands = self.controlMatrix.T.dot(
                            self.slopesBuffer.flatten())
        self.slopesBuffer = numpy.roll(self.slopesBuffer,1,0)
        
        return dmCommands
        
