import numpy as np
import random as rd


def expm(X,t):
    x=0
    li=[i for i in range(t,t-60,-1)]
    for i in range(len(li)):
        x+=np.exp(-i/5)*X[t-i]
    return x/len(X)
def stdexp(X,t):
    dev=0
    for i in range(t-60,t):
        dev+=(np.exp(-i/5)*X[t-i]-expm(X,t))**2
    return dev/len(X)
        
      
import random
def draw(w):
    e=np.random.exponential(w)
    while e<=.01:
        e=np.random.exponential(w)
    return e
def get_timestamp():
    """ Microsecond timestamp """
    return int(1e6 * time.time())
def sum(a,b):
    if a[1]==b[1]:
        return (a[0]+b[0],a[1])
class agent:
    
    
    def __init__(self):
        # set strategy weights
        # set all positive :  this diverges from the paper a little
       
        # random component of spread
        self.pfcast=0
        self.wealth = 0.
        self.bid = 0.
        self.ask = 0.
        self.buy=0
        self.sell=0
        self.ta=500
        self.imp=np.random.uniform(0,1)
        self.l=np.random.randint(10,50)
        self.ul=np.random.randint(85,100)
        self.tol=np.random.uniform(0,2)
        self.pd=np.random.normal(0,.01)
        self.tol1=2
        self.r=[rd.choices([-1,1],k=6) for i in range(2) ]
        self.p=[0 for i in range(2)]
        self.norma=np.random.normal()
        self.wind=np.random.randint(20,50)
        self.w1=np.random.normal(0,1)
        self.w2=np.random.normal(0,1)
        self.ret=0
        self.volat=0
        self.buy=0
        self.ask=0
        self.tw=np.random.randint(10,50)
        # forecast adjustment weight
        
    def updateFcast(self,pric,tau,ret,t,news):
        # weighted forecast value
        
        self.pfcast = (np.mean(pric[t-self.ta:t]))#*(self.w1)+
                                               # np.mean(ret[t-self.l:t])*self.w2)/(self.w1+self.w2)
        #self.pfcast=np.exp(self.pf)*pric[t]
    def updatescor(self,markorder,bs,mu,delt,d1):
      for strtin in range(2):
        ra=self.r[strtin]
        self.p[strtin]=self.p[strtin]-markorder*ra[mu]#+markorder*int(not(ra[mu]))*int(delt>d1)
    def a(self,mu):
      if max(self.p)==min(self.p):
        winn=np.random.randint(0,2)
      else:
        winn=np.argmax(self.p)
      return self.r[winn][mu]
    def calcpara(self,ret,t,sigm):
      rn=(ret[Tinit:t]-np.mean(ret[Tinit:t-1]))/(np.std(ret[Tinit:t-1]))
      if rn[t]>1:
        self.ret=1
      elif rn[t]<-1:
        self.ret=-1
      else:
        self.ret=0
      self.volat=np.sign(sigm/(np.std(ret[Tinit:t-1])))


        # bound the forecast
        
   
  
        
       
        
import numpy as np

class orderBook:
    
    def __init__(self,minP,maxP,deltaP):
        # price ranges on order book
        self.minPrice = minP
        self.maxPrice = maxP
        self.midPrice = price[Tinit]
        # discreteness in book
        self.deltaPrice = deltaP
        # discrete prices
        self.priceVec = np.round(np.arange(minP,maxP+deltaP,deltaP),1)
        self.nPrice = len(self.priceVec)
        # set up lists of lists for bids and asks
        self.bids = []
        self.asks = []
        for i in range(self.nPrice):
            self.bids.append([])
            self.asks.append([])
        # generate best bid
        pmid = self.discretePrice(self.midPrice)
        self.bestBidDex = pmid[0]-2
        self.bestBid = self.realPrice(self.bestBidDex)
        self.bestAskDex = pmid[0]+2 
        self.bestAsk = self.realPrice(self.bestAskDex)
        # drop 5 orders in at best bid
        
       
    # take price and return (index, descrete price)
    def updatebidask(self,price,a,b):
        i = self.nPrice-1
        while self.bids[i]==[]:
            
            i -= 1
            if i==0:
              self.bestBidDex=int(a)
              self.bids[int(a)].append((1,t))
              self.bestBid=self.realPrice(self.bestBidDex)
              i=int(a)
            
        self.bestBidDex = i
        self.bestBid = self.realPrice(i)
        j = 0
        while self.asks[j]==[]:
          
            j += 1
            if j==self.nPrice-1:
              self.bestAskDex=int(b)
              self.asks[int(b)].append((1,t))
              self.bestAsk=self.realPrice(self.bestAskDex)
              j=int(b)
        self.bestAskDex = j
        self.bestAsk = self.realPrice(j)
        
    def discretePrice(self,price):
        iPrice = int((np.round(((price-self.minPrice)/self.deltaPrice),0)))
        iPrice = max(iPrice,0)
        iPrice = min(iPrice,self.nPrice-1)
        
        dPrice = self.minPrice + self.deltaPrice*iPrice
        return (iPrice, dPrice)
    # take discrete price and return real price    
    def realPrice(self,iPrice):
        return self.minPrice + self.deltaPrice*iPrice
    # add bid orders to the book    
    def addlim(self,price,price1,a,b,quant,t,bs,si):
        trade=price1
        qq=(2*np.arctan(si)/np.pi)*3
        x=np.random.exponential(si/(si+10))
        num=np.random.randint(0,2)
        
        #p=p2/(p1+p2)
       # print(dPrice,12,self.bestBid)
        bia=np.random.normal()
        if bs==1:
            ptup=self.discretePrice(price+x+.1)
            iPrice = ptup[0]
            dPrice = ptup[1]
            op=x+.01
            
            self.asks[iPrice].append((1,t))
            
        
            
                #print(a,self.bestBidDex)
                   

                        
            return op

            # push (order,t) onto book at iPrice
            
                        
            # if better than bestBid, then update best
            
        # price > best ask, then execute trade at best ask
        elif bs==-1:
           
            ptup=self.discretePrice(price-x-.1)
            iPrice = ptup[0]
            dPrice = ptup[1]
            op=x+.01
            self.bids[iPrice].append((1,t))
            return op
               # print(b,self.bestAskDex)
        else:
          return 0
         
            
            
                    


    
 
        
        # price < best ask then add to bid side
        
        
        #tradeprice=best ask
        #find out new best ask and update it thats all
        # similarly in addask tradeprice=bestbid and figure out new bid price by manual search
        
        
        
        
        
        
        
            # pop first in order off best ask
        
            # walk up the book to find new best ask
        #print(self.asks[self.bestAskDex],self.bestAsk)

    # cleanse book of old orders   
    def cleanBook(self,t,tau,a,b,pr,g,h):
        # sweep through book
        
        for i in range(0,self.nPrice):
            self.bids[i]=[]
            self.asks[i]=[]
                    
        
        
                    
            #self.bids[i]=[]
            #self.asks[i]=[]
      
        
        
          
       
     
      
        # reset best bid and ask prices
       
        # make sure there is some order at the end
       
    # utility to print the order book 
    def execute(self,tr,liq,pri):
        if liq!=0:
            if ttype==1 and liq>a:
                for i in range(a):
                    trade=self.addmarketAsk(pri,pri,1,t)
            elif ttype==-1 and liq>b:
                for i in range(b):
                    trade=self.addmarketBid(pri,pri,1,t)
            else:
                if ttype==1:
                    for i in range(liq):
                        trade=self.addmarketAsk(pri,pri,1,t)
                else:
                    for i in range(liq):
                        trade=self.addmarketBid(pri,pri,1,t)
            return trade
        else:
            return pri
        
        
    def addmarketBid(self,price,price1,quant,t):
        
       
        tradeInfo = self.asks[self.bestAskDex].pop(0)
        tradePrice = self.bestAsk
        tradeQ     = tradeInfo[0]
        for l in range(self.bestAskDex,self.nPrice):
            if self.asks[l]!=[]:
                break
        if self.asks[l]==[]:
            self.bestAskDex=(self.bestAskDex+1)%self.nPrice
            self.asks[self.bestAskDex].append((1,t))
            
            self.bestAsk=self.priceVec[self.bestAskDex]
        else:
            self.bestAskDex=l
            self.bestAsk=self.priceVec[l]
      
        
        # price < best ask then add to bid side
        
        
        #tradeprice=best ask
        #find out new best ask and update it thats all
        # similarly in addask tradeprice=bestbid and figure out new bid price by manual search
        
        
        
        
        
        
        
            # pop first in order off best ask
        
            # walk up the book to find new best ask
        #print(self.asks[self.bestAskDex],self.bestAsk)

        return tradePrice
    # repeat all this for adding an ask 
    def addmarketAsk(self,price,price1,quant,t):
      
        
       
      
        tradeInfo = self.bids[self.bestBidDex].pop(0)
        tradePrice = self.bestBid
        tradeQ     = tradeInfo[0]
        for k in range(self.bestBidDex,-1,-1):
            if self.bids[k]!=[]:
                break
        if self.bids[k]==[]:
            self.bestBidDex=(self.bestBidDex-1)%self.nPrice
            self.bids[self.bestBidDex].append((1,t))
            
            self.bestBid=self.priceVec[self.bestBidDex]
        else:
            self.bestBidDex=k
            self.bestBid=self.priceVec[k]
        return tradePrice
        
                
                
    def sumorder(self):
        x=0;y=0
        for i in range(self.nPrice):
            if(self.bids[i] != []):
                x+=len(self.bids[i])
        for i in range(self.nPrice):
            if(self.asks[i] != []):
                y+=len(self.asks[i])
        return x,y
        
       
    def printBook(self):
        for i in range(self.nPrice):
            if(self.bids[i] != []):
                print(self.realPrice(i),self.bids[i])
        print("------")
        for i in range(self.nPrice):
            if(self.asks[i] != []):
                print(self.realPrice(i),self.asks[i])
    
    
    
def val(x):
    r = 0
    for i in range(len(x)):
        if (x[i] == 1):
            r = r + 2**(len(x)-i-1)
      
    return r
                            
nAgents =2500
Tinit = 5000
Tmax = 100000
Lmin=5
Lmax=10
pf = 1000.
deltaP = 0.1
price=np.zeros(Tmax+1)
qw=[np.random.choice([1,-1]) for i in range(nAgents)]
price[Tinit]=1000
price[0:Tinit+2]=1000*(1.+0.001*np.random.randn(Tinit+2))
ret   = np.zeros(Tmax+1)
for i in range(1,Tinit):
    ret[i]=np.log(price[i]/price[i-1])
marketBook = orderBook(0,3900.,deltaP) 

voll=[]

volume=np.ones(Tmax)

kMax=.5
agentList = []
# price, return, and volume time series
tota = np.zeros(Tmax+1)

# create agents in list of objects
for i in range(nAgents):
    agentList.append(agent())
tau=10

bid=(1.+0.001*np.random.randn(Tmax))
ask=(1.+0.001*np.random.randn(Tmax))                       
bidd=(1.+0.001*np.random.randn(Tmax))
askk=(1.+0.001*np.random.randn(Tmax))  
meno=np.ones(Tmax)
import numpy.random as rnd
from scipy.stats import norm
import numpy as np

aa,bb=1,1
volatility=[]
volat=[]
bol=np.ones(Tmax)
bol[Tinit-500:Tinit]=550*np.ones(500)
spread=np.ones(Tmax)*5
sol=np.zeros(60)
for t in range(Tinit+1,Tmax):
    # update all forecasts
    tradePrice = price[t]
    LO=np.zeros(nAgents)
    tr=[]
    # draw random agent
    x=np.arange(0,nAgents)
    aa,bb=marketBook.sumorder()
    bid[t]=(marketBook.bestBidDex)
    bidd[t]=marketBook.bestBid
    askk[t]=marketBook.bestAsk
    
    # updating lamda values









    meno[t]=np.mean(bol)
    ask[t]=(marketBook.bestAskDex)
    sigma=np.std([ret[t-105:t]])
    sigma1=np.std(ret[t-60:t])
    volatility.append(sigma)
    volat.append(sigma1)
    med1=np.nanmedian(volat)
    med=np.nanmedian(volatility)
    

    print(sigma/med)
   
    bits = rd.choices([-1,1],k=6)
    actionarray=np.zeros(nAgents)
    if t==Tinit+1:
        st=ret[Tinit-1]
        stt=ret[Tinit]
    else:
        st=np.std(ret[Tinit-1:t-1])
        stt=np.std(ret[Tinit-1:t])
    nr=(ret[Tinit-1:t]-np.mean(ret[Tinit-1:t-1]))/st
    sig=int(np.sign(np.log(np.std(ret[t-105:t])/stt)))
    #for k in range(5):
     # bits[k]=int(np.sign(np.log(np.std(ret[t-10-k*10:t])/np.std(ret[Tinit-1:t-1]))))
    if abs(nr[-1])<1:
      nrr=0
    else:
      nrr=int(np.sign(nr[-1]))
    his=(sig,nrr)
    bits[-1]=nrr
    mu=vall(his)
    
    for xx in range(nAgents):
        
        
       # agentList[xx].updateFcast(price,tau,ret,t,news)

        #agentList[xx].calcpara(ret,t,sigma)
        
        trade=marketBook.addlim(price[t],price[t],bid[t],ask[t],1,t,agentList[xx].a(mu),sigma/med)
        actionarray[xx]=trade
       
        #agentList[xx].updatescor(ttype,actionarray[xx],mu,l1)
                #else:
                   # tr.append('B')
       
               
                #else:
                    #tr.append('S')
            
          
   # for i in range(ll):
    #  lis[i]=np.sum([o[0] for o in (marketBook.bids[i])])
    #  lisask[i]=np.sum([o[0] for o in (marketBook.asks[i])])             
   
      #tr.append(tradePrice)
    # update price and volume
        #print(agentList[i].pfcast)
    
    a,b=marketBook.sumorder()
    
    sol[t%60]=b
    print(a,b,'ab')
    ttype=np.random.choice([1,-1])
    print(mu,ttype,'mu',bits[-1])
    print(spread[t-1],'spr')
    l1=mean(bol[t-501:t-1],spread[t-501:t-1])*(1+np.log(price[t]/np.mean(price[t-500:t])))#*np.exp((-.1*(spread[t-1])))/(np.exp((-(spread[t-1])))+.5)#2*np.arctan(price[t]/np.mean(price[t-1000:t])/np.pi))+1
    l2=mean(bol[t-501:t-1],spread[t-501:t-1])*(1-np.log(price[t]/np.mean(price[t-500:t])))#*np.exp((-.1*(spread[t-1])))/(np.exp((-(spread[t-1])))+.5)##(1-2*np.arctan(price[t]/np.mean(price[t-1000:t])/np.pi))+1
    lambdaa=int(ttype==1)*l1+int(ttype==-1)*l2
    liq=np.random.poisson(lambdaa)
    bol[t]=liq
    tota[t]=liq
    print(lambdaa,'lam',ttype)
    print(liq,a,b)
   
    marketBook.updatebidask(price[t],bid[t],ask[t])
    
    tota[t]=liq
    tradePrice= marketBook.execute(ttype,liq,price[t])
    spread[t]=marketBook.bestAskDex-marketBook.bestBidDex
    print(spread[t])
    if liq!=0 and  t>Tinit+1:
        for x1 in range(nAgents):
            agentList[x1].updatescor(ttype,agentList[x1].a(mu),mu,actionarray[x1],abs(price[t]-tradePrice))
      
   
 
    
   
    print(liq)
    
    #for i in range(ll):
     # lis1[i]=np.sum([o[0] for o in (marketBook.bids[i])])
      #lisask1[i]=np.sum([o[0] for o in (marketBook.asks[i])])
    print(marketBook.bestBid,marketBook.bestAsk)
       
    # set update current forecasts for random agent
    # get demands for random agent
    # potential buyer
    
        
        
        # seller: add ask, or market order
        
    # update price and volume

    # no trade
    price[t+1]=tradePrice
  
    # returns
    ret[t+1]=np.log(price[t+1]/price[t])
    # clear book
    marketBook.cleanBook(t,tau,marketBook.bestBidDex,marketBook.bestAskDex,price[t],bid,ask)
def mean(xx,sprr):
    x=[5/spr for spr in sprr]
    return np.sum([x[i]*xx[i] for i in range(len(xx))])/500
 def vall(x):
    r=0
    for ii in range(0,2):
      r=r+x[1-ii]*2**ii
    if x[0]==1:
      return r+2
    else:
      return r+3
