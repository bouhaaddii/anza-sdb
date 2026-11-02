"""Current model names mapped to immutable archived checkpoint identifiers."""

def archive_model_id(name):
    """Accept CNN as the current name while preserving archived file lookup."""
    key=name.lower().replace('-', '_')
    return 'sdb_cnn' if key=='cnn' else key

def build_cnn(input_size=(9,9,6),dropout_rate=0.3):
    from original_models import build_sdb_cnn
    return build_sdb_cnn(input_size=input_size,dropout_rate=dropout_rate)

def create_model(model_type,input_size=(9,9,6),**kwargs):
    import contextlib,io
    from original_models import create_model as archived_factory
    key=archive_model_id(model_type)
    # Replace the archived console banner while retaining its construction code.
    with contextlib.redirect_stdout(io.StringIO()):
        result=archived_factory(key,input_size=input_size,**kwargs)
    labels={'sdb_cnn':'CNN','pi_cnn':'PI-CNN','cnn_bilstm':'CNN-BiLSTM','fcnn':'FCNN','seabed_net':'SeaBed-Net'}
    print('Creating model: '+labels.get(key,key))
    return result

def prepare_seabed_labels(depths):
    from original_models import prepare_seabed_labels as archived_labels
    return archived_labels(depths)
