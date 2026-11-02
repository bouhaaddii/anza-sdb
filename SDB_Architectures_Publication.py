"""Draw the executed model architectures from inspected Keras layer shapes.

No fitting or checkpoint changes. The source module supplies the preserved
scientific functions; the diagram makes their input/output contracts explicit.
"""
from pathlib import Path
import contextlib,hashlib,importlib.util,io
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

NAMES=['CNN','PI-CNN','CNN-BiLSTM','FCNN','SeaBed-Net']
KEYS=['sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net']
PARAMS=[105377,105441,204193,197057,115845]

def inspect_models(source):
    """Instantiate unchanged definitions and inspect every shape without training."""
    spec=importlib.util.spec_from_file_location('publication_archived_models',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=[]
    for key,name,count in zip(KEYS,NAMES,PARAMS):
        with contextlib.redirect_stdout(io.StringIO()):
            model,loss,weights,metrics=module.create_model(key,input_size=(9,9,6))
        assert model.input_shape==(None,9,9,6)
        assert model.count_params()==count
        layers=[{'type':type(layer).__name__,'output_shape':layer.output_shape} for layer in model.layers]
        conv=[layer for layer in model.layers if type(layer).__name__=='Conv2D']
        pools=[layer for layer in model.layers if type(layer).__name__=='MaxPooling2D']
        assert [x.output_shape[1:] for x in pools[:2]]==[(4,4,32),(2,2,64)]
        assert conv[2].output_shape[1:]==(2,2,128)
        if key=='fcnn':
            assert model.output_shape==(None,9,9,1)
            assert [x.output_shape[1:] for x in model.layers if type(x).__name__=='Conv2DTranspose']==[(4,4,64),(8,8,32)]
        elif key=='seabed_net':
            assert model.output_shape==[(None,1),(None,4)]
            assert weights=={'depth':.7,'seabed_class':.3}
        else:assert model.output_shape==(None,1)
        if key=='pi_cnn':
            assert next(x for x in model.layers if type(x).__name__=='Concatenate').output_shape==(None,129)
        if key=='cnn_bilstm':
            assert next(x for x in model.layers if type(x).__name__=='Reshape').output_shape==(None,1,128)
            assert next(x for x in model.layers if type(x).__name__=='Bidirectional').output_shape==(None,128)
        report.append({'name':name,'input_shape':model.input_shape,'output_shape':model.output_shape,
                       'parameters':count,'layers':layers})
    return {'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'models':report,'scope':'Architecture instantiation and tensor inspection only; no training or inference.'}

def architecture_figure(audit):
    """Compact preparation, shared encoder and five distinct output heads."""
    assert [m['name'] for m in audit['models']]==NAMES
    fig=plt.figure(figsize=(6.56,6.0));ax=fig.add_axes([.006,.01,.988,.98]);ax.set(xlim=(-1,101),ylim=(-2,100));ax.axis('off')
    ink='#172D3D';blue='#E7F1F8';purple='#F1EAF6';amber='#FFF2D9';green='#E8F3F0'
    def txt(x,y,s,size=8.2,bold=False,ha='center'):
        return ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=ink,fontweight='bold' if bold else 'normal',linespacing=1.12)
    def box(x,y,w,h,s,color=blue,size=8.2):
        ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=0.15,rounding_size=.65',facecolor=color,edgecolor='#536879',lw=.65))
        txt(x+w/2,y,s,size)
    def arrow(x1,y1,x2,y2,dashed=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,lw=.8,color=ink,linestyle='--' if dashed else '-'))
    txt(0,97,'(a) From delivered rasters to model inputs',9.2,True,'left')
    box(0,89,23,8,'RGB image\n202 × 210 × 3')
    box(27,89,31,8,'Divide by 10,000; interpolate\n280 × 444 × 6')
    box(62,89,19,8,'Input patch X\n9 × 9 × 6')
    box(84,89,16,8,'Target z\nCentre value',green,7.8)
    arrow(23,89,27,89);arrow(58,89,62,89)
    txt(49,82.8,'Stride 3; retain valid windows. The six input channels are interpolated RGB combinations.',7.7)
    txt(0,77,'(b) Convolutional encoder used by all five models',9.2,True,'left')
    box(0,68,17,9,'Input X\n9 × 9 × 6')
    box(21,68,24,9,'Conv 32 + BN\nPool + dropout\n4 × 4 × 32')
    box(49,68,24,9,'Conv 64 + BN\nPool + dropout\n2 × 2 × 64')
    box(77,68,23,9,'Conv 128 + BN\nFeatures F\n2 × 2 × 128')
    for a,b in [(17,21),(45,49),(73,77)]:arrow(a,68,b,68)
    txt(0,60,'(c) Model-specific heads and evaluated outputs',9.2,True,'left')
    # Each row begins with the same F from panel (b). Abbreviations are in the caption.
    txt(0,54,'CNN',8.6,True,'left')
    box(19,53,20,7,'F → GAP\n128 values')
    box(45,53,28,7,'Dense 64 → 32\nDropout after each')
    box(79,53,21,7,'Linear 1\nScalar output',green)
    arrow(39,53,45,53);arrow(73,53,79,53)
    txt(0,43,'PI-CNN',8.6,True,'left')
    box(19,44,20,5.7,'F → GAP: 128')
    box(19,36.8,20,5.7,'X → feature φ: 1',amber,7.8)
    box(44,41,13,8,'Join\n129',purple)
    box(61,41,18,8,'Dense\n64 → 32')
    box(83,41,17,8,'Linear 1\nScalar',green)
    arrow(39,44,44,42.5);arrow(39,36.8,44,39.5);arrow(57,41,61,41);arrow(79,41,83,41)
    txt(0,29,'CNN-BiLSTM',8.3,True,'left')
    for x,w,s,co in [(19,18,'F → GAP\n128',blue),(41,17,'Reshape\n1 × 128',blue),(62,17,'BiLSTM\n64 + 64',purple),(83,17,'Dense 64, 32\nLinear 1',green)]:box(x,29,w,8,s,co,7.9)
    for a,b in [(37,41),(58,62),(79,83)]:arrow(a,29,b,29)
    txt(0,17,'FCNN',8.6,True,'left')
    for x,w,s,co in [(19,17,'F → TConv\n4 × 4 × 64',blue),(39,17,'TConv\n8 × 8 × 32',blue),(59,19,'Pad + Conv\n9 × 9 × 32',blue),(81,19,'1 × 1 Conv\n9 × 9 × 1',green)]:box(x,17,w,8,s,co,7.9)
    for a,b in [(36,39),(56,59),(78,81)]:arrow(a,17,b,17)
    txt(70,11.5,'81 repeated centre labels in training; evaluate the centre output.',7.2)
    txt(0,5,'SeaBed-Net',8.6,True,'left')
    box(19,5,18,8,'F → GAP\n128')
    box(42,7.7,27,4.7,'Dense 64 → 32',blue,7.8)
    box(42,1.6,27,4.7,'Dense 64 → 32',purple,7.8)
    box(75,7.7,25,4.7,'Linear 1: scalar',green,7.8)
    box(75,1.6,25,4.7,'Softmax 4: classes',purple,7.8)
    arrow(37,5,42,7.7);arrow(37,5,42,1.6);arrow(69,7.7,75,7.7);arrow(69,1.6,75,1.6)
    return fig

def evaluation_figure(counts):
    """Show the observed training, selection and evaluation roles separately."""
    assert set(counts)=={'train','val','test'}
    fig=plt.figure(figsize=(6.56,2.9));ax=fig.add_axes([.01,.02,.98,.96]);ax.set(xlim=(-1,101),ylim=(-2,100));ax.axis('off')
    def box(x,y,w,h,text,colour):
        ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=.15,rounding_size=1',facecolor=colour,edgecolor='#536879',lw=.7))
        ax.text(x+w/2,y,text,ha='center',va='center',fontsize=8.1,linespacing=1.2,color='#172D3D')
    def arrow(x,y,a,b):ax.add_patch(FancyArrowPatch((x,y),(a,b),arrowstyle='-|>',mutation_scale=8,lw=.9,color='#172D3D'))
    box(0,80,23,24,f"Training\n{counts['train']:,} centres",'#E7F1F8')
    box(32,80,31,24,'Fit each architecture\nAdam; batch 512\nUp to 300 epochs','#E7F1F8')
    box(72,80,28,24,'Record loss curves\nTraining + validation','#E8F3F0')
    box(0,46,23,24,f"Validation\n{counts['val']:,} centres",'#FFF2D9')
    box(32,46,31,24,'Monitor validation loss\nEarly stopping\nLearning-rate reduction','#FFF2D9')
    box(72,46,28,24,'Select checkpoint\nMinimum validation loss','#E8F3F0')
    box(0,12,23,24,f"Test\n{counts['test']:,} centres",'#F1EAF6')
    box(32,12,31,24,'Evaluate selected weights\nSame test centres\nfor all five models','#F1EAF6')
    box(72,12,28,24,'Paired predictions\nMetrics and spatial maps','#E8F3F0')
    arrow(23,80,32,80);arrow(63,80,72,80);arrow(23,46,32,46)
    arrow(47.5,68,47.5,58);arrow(63,46,72,46)
    arrow(23,12,32,12);arrow(72,40,61,24);arrow(63,12,72,12)
    return fig
