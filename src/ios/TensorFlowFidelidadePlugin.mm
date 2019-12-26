//
//  TensorFlowFidelidadePlugin.mm
//
//  Created by Andre Grillo on 15/12/2019.
//  Copyright © 2019 Andre Grillo. All rights reserved.
//

#import <TensorIO/TensorIO-umbrella.h>
#import <Cordova/CDV.h>

@interface TensorFlowFidelidadePlugin: CDVPlugin {
    id model;
    UIImage *image;
    CDVPluginResult *pluginResult;
}

@property (strong, nonatomic) CDVInvokedUrlCommand* commandHelper;
- (void)loadModel:(CDVInvokedUrlCommand*)command;

@end

@implementation TensorFlowFidelidadePlugin

- (void)loadModel:(CDVInvokedUrlCommand*)command {
    
    self.commandHelper = command;
    image = [self decodeBase64ToImage:[command.arguments objectAtIndex:1]];
    [command.arguments objectAtIndex:1];
    NSString *modelName = [command.arguments objectAtIndex:0];
    NSString *path = [NSBundle.mainBundle bundlePath];
    NSError *error;
    
    // Checks the model index on ModelBundleManager Array
    int modelIndex = 3;
    [TIOModelBundleManager.sharedManager loadModelBundlesAtPath:path error:&error];
    for (int i = 0; i < 3; i++) {
        if ([modelName isEqualToString:TIOModelBundleManager.sharedManager.modelBundles[i].identifier]) {
            modelIndex = i;
        }
    }
    
    //Checks if the modelIndex was found
    if (modelIndex > 2) {
        NSLog(@"Error: Model not found.");
        return;
    }
    
    TIOModelBundle *bundle = TIOModelBundleManager.sharedManager.modelBundles[modelIndex];
    model = [bundle newModel];
    
    UIImage *resizedImage;
    
    if ([modelName isEqualToString:@"enq_model"]){
        //Processa tamanho da imagem (resize)
        if (image.size.width == 64.0 && image.size.height == 64.0) {
            [self runModelEnquadramento:image];
        } else {
            CGSize modelImageSize = CGSizeMake(64.0, 64.0);
            resizedImage = [self resizeImage:image tosize:(modelImageSize)];
            [self runModelEnquadramento:resizedImage];
        }
    }
    else if ([modelName isEqualToString:@"quality_model"]){
        //Processa tamanho da imagem (resize)
        if (image.size.width == 64.0 && image.size.height == 64.0) {
            [self runModelEnquadramento:image];
        } else {
            CGSize modelImageSize = CGSizeMake(224.0, 224.0);
            resizedImage = [self resizeImage:image tosize:(modelImageSize)];
            [self runModelQuality:resizedImage];
        }
    }
    else if ([modelName isEqualToString:@"unet_vehicle_model"]){
        //Processa tamanho da imagem (resize)
        if (image.size.width == 224.0 && image.size.height == 224.0) {
            [self runModelunet_vehicle_model];
        } else {
            CGSize modelImageSize = CGSizeMake(224.0, 224.0);
            resizedImage = [self resizeImage:image tosize:(modelImageSize)];
            [self runModelunet_vehicle_model];
        }
    }
}

- (UIImage *)decodeBase64ToImage:(NSString *)strEncodeData {
  NSData *data = [[NSData alloc]initWithBase64EncodedString:strEncodeData options:NSDataBase64DecodingIgnoreUnknownCharacters];
  return [UIImage imageWithData:data];
}

- (UIImage *)resizeImage:(UIImage *)image tosize:(CGSize)newSize {
    //UIGraphicsBeginImageContext(newSize);
    // In next line, pass 0.0 to use the current device's pixel scaling factor (and thus account for Retina resolution).
    // Pass 1.0 to force exact pixel size.
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

- (void)runModelEnquadramento:(UIImage *)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
        NSDictionary *inference = (NSDictionary *)[self->model runOn:buffer];
        NSLog(@"INFERENCE: %@",inference);
        NSDictionary<NSString*,NSNumber*> *classification = inference[@"output"];
        __block NSString *highKey;
        __block NSNumber *highVal;
        [classification enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSNumber *val, BOOL *stop) {
            if (highVal == nil || [val compare:highVal] == NSOrderedDescending) {
                highKey = key;
                highVal = val;
            }
        }];
        
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:highKey];
        [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
        //NSLog(@"%@: %@".capitalizedString, highKey, highVal);
        //self.label.text = [NSString stringWithFormat:@"%@: %@", highKey, highVal];
    });
}

- (void)runModelQuality:(UIImage *)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
        NSDictionary *inference = (NSDictionary *)[self->model runOn:buffer];
        NSArray *result = inference[@"output"];
        NSLog(@"%@",result);
        if ([result[0] floatValue] > [result[1] floatValue]) {
            NSLog(@"QUALIDADE RUIM! Valor da inferência: %@",result[0]);
            pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"false"];
            [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
        } else {
            NSLog(@"QUALIDADE BOA! Valor da inferência: %@",result[1]);
            pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"true"];
            [self.commandDelegate sendPluginResult:pluginResult callbackId:self.commandHelper.callbackId];
        }
    });
}

- (void)runModelunet_vehicle_model{
    dispatch_async(dispatch_get_main_queue(), ^{
        dispatch_async(dispatch_get_main_queue(), ^{
            TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];
            NSDictionary<TIOData> *resultDict = (NSDictionary *)[self->model runOn:buffer];
            TIOPixelBuffer *resultBuffer = resultDict[@"output"];
            CVPixelBufferRef pixelBuffer = resultBuffer.pixelBuffer;
            
            //Para ver que o pixelBuffer está no formato float32
            //NSLog(@"tipo: %u",(unsigned int)CVPixelBufferGetPixelFormatType(pixelBuffer));
            
            UIImage *resultImage = [self uiImageFromPixelBuffer:pixelBuffer];
//            NSLog(@"Foto: %@",pixelBuffer);
            
            [self logPixelsOfImage:resultImage];
            
//            [self checkPixelBuffer:pixelBuffer];
            

        });
    });
}

- (UIImage*)uiImageFromPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    CIImage *ciImage = [CIImage imageWithCVPixelBuffer:pixelBuffer];

    CIContext *temporaryContext = [CIContext contextWithOptions:nil];
    CGImageRef videoImage = [temporaryContext
                       createCGImage:ciImage
                       fromRect:CGRectMake(0, 0,
                              CVPixelBufferGetWidth(pixelBuffer),
                              CVPixelBufferGetHeight(pixelBuffer))];

    UIImage *uiImage = [UIImage imageWithCGImage:videoImage];
    CGImageRelease(videoImage);
    return uiImage;
}

- (void)checkPixelBuffer:(CVPixelBufferRef)pixelBuffer{
//    CVPixelBufferRef pixelBuffer = _lastDepthData.depthDataMap;

    CVPixelBufferLockBaseAddress(pixelBuffer, 0);

    size_t cols = CVPixelBufferGetWidth(pixelBuffer);
    size_t rows = CVPixelBufferGetHeight(pixelBuffer);
    
    NSLog(@"Valor de cols: %lu, Valor de rows: %lu", cols, rows);
    
    Float32 *baseAddress = (Float32 *)(CVPixelBufferGetBaseAddress(pixelBuffer));

    // This next step is not necessary, but I include it here for illustration,
    // you can get the type of pixel format, and it is associated with a kCVPixelFormatType
    // this can tell you what type of data it is e.g. in this case Float32

//    OSType type = CVPixelBufferGetPixelFormatType(pixelBuffer);
//
//    if (type != kCVPixelFormatType_DepthFloat32) {
//        NSLog(@"Wrong type");
//    }

    // Arbitrary values of x and y to sample
    int x = 20; // must be lower that cols
    int y = 30; // must be lower than rows

    // Get the pixel.  You could iterate here of course to get multiple pixels!
    int baseAddressIndex = y  * (int)cols + x;
    const Float32 pixel = baseAddress[baseAddressIndex];
    
    NSLog(@"Valor do Pixel: %f", pixel);

    CVPixelBufferUnlockBaseAddress( pixelBuffer, 0 );
}

- (void)logPixelsOfImage:(UIImage*)image {
    // 1. Get pixels of image
    CGImageRef inputCGImage = [image CGImage];
    NSUInteger width = CGImageGetWidth(inputCGImage);
    NSUInteger height = CGImageGetHeight(inputCGImage);
    
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    
    UInt32 * pixels;
    pixels = (UInt32 *) calloc(height * width, sizeof(UInt32));
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pixels, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast|kCGBitmapByteOrder32Big);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), inputCGImage);
    
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    
#define Mask8(x) ( (x) & 0xFF )
#define R(x) ( Mask8(x) )
#define G(x) ( Mask8(x >> 8 ) )
#define B(x) ( Mask8(x >> 16) )
    
    // 2. Iterate and log!
    NSLog(@"Brightness of image:");
    UInt32 * currentPixel = pixels;
    for (NSUInteger j = 0; j < height; j++) {
        for (NSUInteger i = 0; i < width; i++) {
            //        printf("%i ",currentPixel);
            UInt32 color = *currentPixel;
            //      printf("%i ",color);
            //      printf("R: %i, G: %i, B: %i - ", R(color), G(color), B(color));
            printf("%3.0f ", (R(color)+G(color)+B(color))/3.0);
            currentPixel++;
        }
        printf("\n");
    }
    
    free(pixels);
    
#undef R
#undef G
#undef B
    
}

@end
