import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

class SDBDataPreprocessor:
    """
    Classe de prétraitement pour les données SDB (Satellite-Derived Bathymetry).
    """
    
    def __init__(self, data_dir):
        """
        Initialise le prétraitement.
        
        Args:
            data_dir (str): Chemin du dossier contenant les données
        """
        self.data_dir = data_dir
        self.multispectral_path = os.path.join(data_dir, "1.tif")
        self.bathymetric_path = os.path.join(data_dir, "2.tif")
        
        # Paramètres standards pour SDB CNN
        self.target_height = 280      # Hauteur standard
        self.target_width = 444       # Largeur standard
        self.target_bands_ms = 6      # Bandes pour multispectral
        self.target_bands_depth = 1   # Bandes pour bathymétrie
        
    def analyze_input_data(self):
        """
        Analyse les données d'entrée.
        
        Returns:
            dict: Informations sur les images
        """
        print("="*80)
        print("ANALYSE DES DONNEES D'ENTREE")
        print("="*80)
        
        info = {}
        
        try:
            # Analyser l'image multispectrale
            with rasterio.open(self.multispectral_path) as src:
                info['multispectral'] = {
                    'filename': os.path.basename(self.multispectral_path),
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'resolution': src.res,
                    'bounds': src.bounds
                }
                
                # Lire les données pour statistiques
                data = src.read()
                info['multispectral']['stats'] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                
                print(f"\n📊 IMAGE MULTISPECTRALE (1.tif):")
                print(f"   • Dimensions: {src.width} × {src.height}")
                print(f"   • Bandes: {src.count}")
                print(f"   • Type: {src.dtypes[0]}")
                print(f"   • CRS: {src.crs}")
                print(f"   • Résolution: {src.res[0]:.1f}m")
                print(f"   • Plage: {data.min():.2f} à {data.max():.2f}")
        
        except Exception as e:
            print(f"❌ Erreur analyse multispectrale: {e}")
            return None
        
        try:
            # Analyser l'image bathymétrique
            with rasterio.open(self.bathymetric_path) as src:
                info['bathymetric'] = {
                    'filename': os.path.basename(self.bathymetric_path),
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'resolution': src.res,
                    'bounds': src.bounds
                }
                
                # Lire les données pour statistiques
                data = src.read()
                info['bathymetric']['stats'] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                
                print(f"\n📊 IMAGE BATHYMETRIQUE (2.tif):")
                print(f"   • Dimensions: {src.width} × {src.height}")
                print(f"   • Bandes: {src.count}")
                print(f"   • Type: {src.dtypes[0]}")
                print(f"   • CRS: {src.crs}")
                print(f"   • Résolution: {src.res[0]:.1f}m")
                print(f"   • Plage: {data.min():.2f} à {data.max():.2f}")
        
        except Exception as e:
            print(f"❌ Erreur analyse bathymétrique: {e}")
            return None
        
        return info
    
    def normalize_multispectral(self, data):
        """
        Normalise les données multispectrales.
        
        Args:
            data (numpy.ndarray): Données multispectrales
            
        Returns:
            numpy.ndarray: Données normalisées
        """
        # Convertir en float32 si nécessaire
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Pour Sentinel-2 L2A, les valeurs sont généralement entre 0 et 1
        # Si les valeurs sont > 1, on suppose qu'elles sont en DN*10000
        if data.max() > 1.0:
            print(f"   Normalisation: division par 10000")
            data = data / 10000.0
        
        # Clip les valeurs entre 0 et 1 (pour éliminer les outliers)
        data = np.clip(data, 0, 1)
        
        return data
    
    def normalize_bathymetric(self, data):
        """
        Normalise les données bathymétriques.
        
        Args:
            data (numpy.ndarray): Données bathymétriques
            
        Returns:
            numpy.ndarray: Données normalisées
        """
        # Convertir en float32
        data = data.astype(np.float32)
        
        # Pour les données bathymétriques, on peut normaliser entre -1 et 1
        # ou garder les valeurs réelles selon l'application
        print(f"   Conservation des valeurs réelles (pas de normalisation)")
        
        return data
    
    def resize_image(self, src_path, target_height, target_width, target_bands, 
                     is_multispectral=True, output_path=None):
        """
        Redimensionne une image aux dimensions cibles.
        
        Args:
            src_path (str): Chemin de l'image source
            target_height (int): Hauteur cible
            target_width (int): Largeur cible
            target_bands (int): Nombre de bandes cible
            is_multispectral (bool): True si image multispectrale
            output_path (str): Chemin de sortie
            
        Returns:
            tuple: (données redimensionnées, métadonnées)
        """
        try:
            with rasterio.open(src_path) as src:
                # Lire les données
                data = src.read()
                
                print(f"\n🔧 PRETRAITEMENT DE {os.path.basename(src_path)}:")
                print(f"   • Dimensions originales: {src.height} × {src.width}")
                print(f"   • Dimensions cibles: {target_height} × {target_width}")
                print(f"   • Bandes originales: {data.shape[0]}")
                print(f"   • Bandes cibles: {target_bands}")
                
                # Normalisation
                if is_multispectral:
                    data = self.normalize_multispectral(data)
                else:
                    data = self.normalize_bathymetric(data)
                
                # Redimensionnement avec interpolation bilinéaire
                print(f"   • Redimensionnement en cours...")
                from scipy.ndimage import zoom
                
                # Calculer les facteurs de zoom
                zoom_factors = (
                    target_bands / data.shape[0],
                    target_height / data.shape[1],
                    target_width / data.shape[2]
                )
                
                # Redimensionner
                resized_data = zoom(data, zoom_factors, order=1)  # order=1 pour bilinéaire
                
                # Si le redimensionnement a créé des bandes supplémentaires par duplication
                # (c'est normal avec zoom), on s'assure d'avoir exactement target_bands
                if resized_data.shape[0] > target_bands:
                    resized_data = resized_data[:target_bands, :, :]
                
                # Mettre à jour les métadonnées
                profile = src.profile.copy()
                
                # Calculer la nouvelle transformation
                scale_x = target_width / src.width
                scale_y = target_height / src.height
                transform = src.transform * Affine.scale(1/scale_x, 1/scale_y)
                
                profile.update({
                    'height': target_height,
                    'width': target_width,
                    'count': target_bands,
                    'dtype': resized_data.dtype,
                    'transform': transform,
                    'crs': src.crs,  # Conserver le CRS original
                    'compress': 'lzw'
                })
                
                if not is_multispectral:
                    profile['nodata'] = -9999.0
                
                # Sauvegarder si un chemin de sortie est fourni
                if output_path:
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(resized_data)
                    
                    print(f"   ✓ Image sauvegardée: {os.path.basename(output_path)}")
                
                return resized_data, profile
                
        except Exception as e:
            print(f"   ❌ Erreur redimensionnement: {e}")
            raise
    
    def preprocess_for_sdb_cnn(self):
        """
        Prétraite les données pour le modèle SDB CNN.
        
        Returns:
            dict: Informations sur le prétraitement
        """
        print("\n" + "="*80)
        print("PRETRAITEMENT POUR SDB CNN")
        print("="*80)
        
        # Analyser les données d'entrée
        input_info = self.analyze_input_data()
        if not input_info:
            return None
        
        # Chemins de sortie
        output_ms = os.path.join(self.data_dir, "ANZA_rgb.tif")
        output_depth = os.path.join(self.data_dir, "ANZA_crt.tif")
        
        # Vérifier si les fichiers de sortie existent déjà
        if os.path.exists(output_ms) or os.path.exists(output_depth):
            print("\n⚠  ATTENTION: Fichiers de sortie existants détectés!")
            overwrite = input("Voulez-vous les écraser? (O/N): ").strip().upper()
            if overwrite not in ['O', 'OUI', 'Y', 'YES']:
                print("Prétraitement annulé.")
                return None
        
        result = {
            'input_info': input_info,
            'output_files': {},
            'preprocessing_params': {
                'target_height': self.target_height,
                'target_width': self.target_width,
                'target_bands_ms': self.target_bands_ms,
                'target_bands_depth': self.target_bands_depth
            }
        }
        
        try:
            # Prétraitement de l'image multispectrale
            print("\n" + "-"*80)
            print("PRETRAITEMENT MULTISPECTRAL")
            print("-"*80)
            
            ms_data, ms_profile = self.resize_image(
                self.multispectral_path,
                self.target_height,
                self.target_width,
                self.target_bands_ms,
                is_multispectral=True,
                output_path=output_ms
            )
            
            result['output_files']['multispectral'] = {
                'path': output_ms,
                'data_shape': ms_data.shape,
                'dtype': str(ms_data.dtype),
                'profile': ms_profile
            }
            
            # Prétraitement de l'image bathymétrique
            print("\n" + "-"*80)
            print("PRETRAITEMENT BATHYMETRIQUE")
            print("-"*80)
            
            depth_data, depth_profile = self.resize_image(
                self.bathymetric_path,
                self.target_height,
                self.target_width,
                self.target_bands_depth,
                is_multispectral=False,
                output_path=output_depth
            )
            
            result['output_files']['bathymetric'] = {
                'path': output_depth,
                'data_shape': depth_data.shape,
                'dtype': str(depth_data.dtype),
                'profile': depth_profile
            }
            
            # Vérification finale
            print("\n" + "-"*80)
            print("VERIFICATION FINALE")
            print("-"*80)
            
            self.verify_preprocessing(result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Erreur lors du prétraitement: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_preprocessing(self, result):
        """
        Vérifie la qualité du prétraitement.
        
        Args:
            result (dict): Résultat du prétraitement
        """
        try:
            print("✓ Vérification des dimensions...")
            
            ms_shape = result['output_files']['multispectral']['data_shape']
            depth_shape = result['output_files']['bathymetric']['data_shape']
            
            # Vérifier les dimensions
            if ms_shape[1:] == depth_shape[1:]:  # Ignorer le nombre de bandes
                print(f"  ✓ Dimensions compatibles: {ms_shape[1]} × {ms_shape[2]}")
            else:
                print(f"  ❌ Dimensions incompatibles!")
                print(f"    • Multispectral: {ms_shape[1]} × {ms_shape[2]}")
                print(f"    • Bathymétrique: {depth_shape[1]} × {depth_shape[2]}")
            
            # Vérifier le CRS
            ms_crs = result['output_files']['multispectral']['profile']['crs']
            depth_crs = result['output_files']['bathymetric']['profile']['crs']
            
            if ms_crs == depth_crs:
                print(f"  ✓ Même système de projection: {ms_crs}")
            else:
                print(f"  ⚠  Systèmes de projection différents")
                print(f"    • Multispectral: {ms_crs}")
                print(f"    • Bathymétrique: {depth_crs}")
            
            # Vérifier les statistiques
            print("\n✓ Statistiques des images prétraitées:")
            
            with rasterio.open(result['output_files']['multispectral']['path']) as src:
                ms_data = src.read()
                print(f"\n  Multispectral ({os.path.basename(src.name)}):")
                print(f"    • Min: {ms_data.min():.4f}")
                print(f"    • Max: {ms_data.max():.4f}")
                print(f"    • Mean: {ms_data.mean():.4f}")
                print(f"    • Std: {ms_data.std():.4f}")
            
            with rasterio.open(result['output_files']['bathymetric']['path']) as src:
                depth_data = src.read()
                print(f"\n  Bathymétrique ({os.path.basename(src.name)}):")
                print(f"    • Min: {depth_data.min():.4f}")
                print(f"    • Max: {depth_data.max():.4f}")
                print(f"    • Mean: {depth_data.mean():.4f}")
                print(f"    • Std: {depth_data.std():.4f}")
            
            print("\n✓ Vérification terminée avec succès!")
            
        except Exception as e:
            print(f"❌ Erreur vérification: {e}")
    
    def generate_summary_report(self, result):
        """
        Génère un rapport de prétraitement.
        
        Args:
            result (dict): Résultat du prétraitement
        """
        print("\n" + "="*80)
        print("RAPPORT DE PRETRAITEMENT")
        print("="*80)
        
        if not result:
            print("Aucun résultat à afficher.")
            return
        
        print("\n📋 PARAMETRES DE PRETRAITEMENT:")
        print("-"*40)
        params = result['preprocessing_params']
        print(f"  • Hauteur cible: {params['target_height']}px")
        print(f"  • Largeur cible: {params['target_width']}px")
        print(f"  • Bandes multispectrales: {params['target_bands_ms']}")
        print(f"  • Bandes bathymétriques: {params['target_bands_depth']}")
        
        print("\n📁 FICHIERS GENERES:")
        print("-"*40)
        for file_type, file_info in result['output_files'].items():
            print(f"\n  {file_type.upper()}:")
            print(f"    • Fichier: {os.path.basename(file_info['path'])}")
            print(f"    • Dimensions: {file_info['data_shape']}")
            print(f"    • Type: {file_info['dtype']}")
        
        print("\n🎯 CONSEILS POUR L'UTILISATION AVEC SDB CNN:")
        print("-"*40)
        print("1. Utilisez les images prétraitées comme entrée du modèle")
        print("2. Vérifiez que les dimensions correspondent aux attentes du modèle")
        print("3. Assurez-vous que les valeurs sont normalisées correctement")
        print("4. Pour l'entraînement, vous devrez probablement diviser les données")
        print("5. Considérez l'augmentation de données pour améliorer la robustesse")
        
        print("\n🔧 PROCHAINES ETAPES:")
        print("-"*40)
        print("1. Charger les images prétraitées dans votre script Python")
        print("2. Préparer les paires (multispectral, bathymétrie) pour l'entraînement")
        print("3. Diviser en ensembles train/val/test")
        print("4. Configurer et entraîner le modèle CNN")
        print("5. Évaluer les performances")
        
        print("\n" + "="*80)

def install_required_packages():
    """Installe les packages requis si nécessaire."""
    required = ['rasterio', 'numpy', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("📦 Installation des packages requis...")
        import subprocess
        import sys
        
        for package in missing:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✓ Tous les packages sont installés!")

def main():
    """
    Fonction principale.
    """
    print("="*80)
    print("PRETRAITEUR DE DONNEES SDB")
    print("="*80)
    print("Prétraitement des images pour Satellite-Derived Bathymetry CNN")
    print("="*80)
    
    # Installer les packages requis
    install_required_packages()
    
    # Définir le dossier de données
    data_dir = os.environ.get("ANZA_RAW_DATA", os.path.dirname(os.path.abspath(__file__)))
    
    # Vérifier que le dossier existe
    if not os.path.exists(data_dir):
        print(f"❌ Dossier introuvable: {data_dir}")
        return
    
    # Vérifier que les fichiers nécessaires existent
    required_files = ["1.tif", "2.tif"]
    missing_files = []
    
    print("\n🔍 Vérification des fichiers...")
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (manquant)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {', '.join(missing_files)}")
        print(f"Assurez-vous que les fichiers sont dans: {data_dir}")
        return
    
    # Initialiser le prétraitement
    preprocessor = SDBDataPreprocessor(data_dir)
    
    # Lancer le prétraitement
    print("\n" + "="*80)
    print("LANCEMENT DU PRETRAITEMENT")
    print("="*80)
    
    result = preprocessor.preprocess_for_sdb_cnn()
    
    # Générer le rapport
    if result:
        preprocessor.generate_summary_report(result)
        
        print("\n" + "="*80)
        print("✅ PRETRAITEMENT TERMINE AVEC SUCCES!")
        print("="*80)
        print("\nLes images sont maintenant prêtes pour le modèle SDB CNN.")
        print(f"\nFichiers générés dans: {data_dir}")
        print("  1. ANZA_rgb.tif - Image multispectrale prétraitée")
        print("  2. ANZA_crt.tif - Image bathymétrique prétraitée")
    else:
        print("\n" + "="*80)
        print("❌ PRETRAITEMENT ECHOUE")
        print("="*80)
        print("\nVeuillez vérifier les erreurs ci-dessus.")
    
    print("\nAppuyez sur Entrée pour quitter...")
    input()

if __name__ == "__main__":
    main()